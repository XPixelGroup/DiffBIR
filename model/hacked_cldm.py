from typing import Mapping, Any, Dict, Optional, Tuple
import copy
from collections import OrderedDict

import ldm.modules.diffusionmodules.model as model_module
from model.cldm import ControlLDM
from model.cldm import ControlledUnetModel
from model.cldm import ControlNet
from model.spaced_sampler import SpacedSampler
from model.cond_fn import Guidance
from utils.image import (
    wavelet_reconstruction, adaptive_instance_normalization
)
from ldm.modules.attention import CrossAttention, FeedForward, BasicTransformerBlock
from ldm.modules.diffusionmodules.model import AttnBlock, MemoryEfficientCrossAttentionWrapper, MemoryEfficientAttnBlock
from ldm.modules.diffusionmodules.util import (
    timestep_embedding,
    GroupNorm32
)
from ldm.util import exists, instantiate_from_config, default
from utils.common import frozen_module

import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
import numpy as np
from tqdm import tqdm


try:
    import xformers
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


class DeepCacheConfig:
    enabled = False
    cache_block_index = 0
    cache_interval = 5
    uniform = False
    pow = 1.4
    center = 15


class OptimizationFlag:
    xformers = False
    sdp_attn = False
    autocast = False
    fp16 = False

    deepcache = DeepCacheConfig()

    @classmethod
    def enable_autocast(cls) -> None:
        cls.autocast = True
    
    @classmethod
    def is_autocast_enabled(cls) -> bool:
        return cls.autocast

    @classmethod
    def disable_autocast(cls) -> bool:
        cls.autocast = False

    @classmethod
    def enable_fp16(cls) -> None:
        cls.fp16 = True

    @classmethod
    def is_fp16_enabled(cls) -> bool:
        return cls.fp16

    @classmethod
    def disable_fp16(cls) -> None:
        cls.fp16 = False
    
    @classmethod
    def enable_xformers(cls) -> None:
        assert XFORMERS_IS_AVAILBLE, "xformers is not available on your current environment"
        cls.xformers = True

    @classmethod
    def is_xformers_enabled(cls) -> bool:
        return cls.xformers
    
    @classmethod
    def enable_deepcache(cls, cache_block_index, cache_interval, uniform, pow, center) -> None:
        cls.deepcache.enabled = True
        cls.deepcache.cache_block_index = cache_block_index
        cls.deepcache.cache_interval = cache_interval
        cls.deepcache.uniform = uniform
        cls.deepcache.pow = pow
        cls.deepcache.center = center

    @classmethod
    def is_deepcache_enabled(cls) -> bool:
        return cls.deepcache.enabled
    
    @classmethod
    def disable_deepcache(cls) -> None:
        cls.deepcache.enabled = False


def hacked_attention_forward(self, x, context=None, mask=None):
    h = self.heads

    q = self.to_q(x)
    context = default(context, x)
    k = self.to_k(context)
    v = self.to_v(context)

    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

    sim = einsum('b i d, b j d -> b i j', q, k) * self.scale
    
    del q, k

    if exists(mask):
        mask = rearrange(mask, 'b ... -> b (...)')
        max_neg_value = -torch.finfo(sim.dtype).max
        mask = repeat(mask, 'b j -> (b h) () j', h=h)
        sim.masked_fill_(~mask, max_neg_value)

    # attention, what we cannot get enough of
    sim = sim.softmax(dim=-1)

    out = einsum('b i j, b j d -> b i d', sim, v)
    out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
    return self.to_out(out)


def hacked_unet_forward(self, x, timesteps=None, context=None, control=None, only_mid_control=False, prev_feat=None):
    dtype = torch.float16 if OptimizationFlag.is_fp16_enabled() else torch.float32
    hs = []
    use_cache = prev_feat is not None

    with torch.no_grad():
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        t_emb = t_emb.type(dtype)
        emb = self.time_embed(t_emb)
        h = x.type(dtype)
        context = context.type(dtype)
        for i, module in enumerate(self.input_blocks):
            if use_cache:
                if i > OptimizationFlag.deepcache.cache_block_index:
                    break
            h = module(h, emb, context)
            hs.append(h)
        if not use_cache:
            h = self.middle_block(h, emb, context)

    if not use_cache:
        if control is not None:
            h += control.pop()
    else:
        # pop useless control (will be deprecated if control has also been cached)
        control.pop()
    
    for i, module in enumerate(self.output_blocks):
        if use_cache:
            if i < len(self.output_blocks) - OptimizationFlag.deepcache.cache_block_index - 1:
                # pop useless control
                control.pop()
                continue
            elif i == len(self.output_blocks) - OptimizationFlag.deepcache.cache_block_index - 1:
                h = prev_feat

        if OptimizationFlag.is_deepcache_enabled():
            if i == len(self.output_blocks) - OptimizationFlag.deepcache.cache_block_index - 1:
                self.cache_feat = h

        if only_mid_control or control is None:
            h = torch.cat([h, hs.pop()], dim=1)
        else:
            h = torch.cat([h, hs.pop() + control.pop()], dim=1)
        h = module(h, emb, context)

    # h = h.type(x.dtype)
    return self.out(h).type(x.dtype)


def hacked_controlnet_forward(self, x, hint, timesteps, context, **kwargs):
    dtype = torch.float16 if OptimizationFlag.is_fp16_enabled() else torch.float32
    t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
    t_emb = t_emb.type(dtype)
    emb = self.time_embed(t_emb)
    x = torch.cat((x, hint), dim=1)
    outs = []
    context = context.type(dtype)

    h = x.type(dtype)
    for module, zero_conv in zip(self.input_blocks, self.zero_convs):
        h = module(h, emb, context)
        outs.append(zero_conv(h, emb, context))

    h = self.middle_block(h, emb, context)
    outs.append(self.middle_block_out(h, emb, context))

    return outs


def convert_module_to_f32(l):
    """
    Convert primitive modules to float32.
    """
    if isinstance(l, GroupNorm32):
        l.float()


def hacked_cldm_to_fp16(self) -> None:
    print(f"hacked_cldm_to_fp16: enable fp16 mode, convert unet and controlnet to fp16")
    self.model.diffusion_model.half()
    self.model.diffusion_model.apply(convert_module_to_f32)
    self.control_model.half()
    self.control_model.apply(convert_module_to_f32)
    self.model.diffusion_model.dtype = torch.float16
    self.control_model.dtype = torch.float16


def hacked_cldm_to_fp32(self) -> None:
    print(f"hacked_cldm_to_fp32: enable fp32 mode, convert unet and controlnet to fp32")
    self.float()
    self.model.diffusion_model.dtype = torch.float32
    self.control_model.dtype = torch.float32


def hacked_cldm_apply_model(self, x_noisy, t, cond, prev_feat):
    assert isinstance(cond, dict)
    diffusion_model = self.model.diffusion_model

    cond_txt = torch.cat(cond['c_crossattn'], 1)

    if cond['c_latent'] is None:
        eps = diffusion_model(
            x=x_noisy, timesteps=t, context=cond_txt, control=None, only_mid_control=self.only_mid_control,
            prev_feat=prev_feat
        )
    else:
        control = self.control_model(
            x=x_noisy, hint=torch.cat(cond['c_latent'], 1),
            timesteps=t, context=cond_txt
        )
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        eps = diffusion_model(
            x=x_noisy, timesteps=t, context=cond_txt, control=control, only_mid_control=self.only_mid_control,
            prev_feat=prev_feat
        )

    return eps


def hacked_sampler_predict_noise(
    self,
    x: torch.Tensor,
    t: torch.Tensor,
    cond: Dict[str, torch.Tensor],
    cfg_scale: float,
    uncond: Optional[Dict[str, torch.Tensor]]
) -> torch.Tensor:
    if uncond is None or cfg_scale == 1.:
        model_output = self.model.apply_model(x, t, cond, self.prev_feat_cond)
        self.cache_feat_cond = self.model.model.diffusion_model.cache_feat
        self.cache_feat_uncond = None
    else:
        # apply classifier-free guidance
        model_cond = self.model.apply_model(x, t, cond, self.prev_feat_cond)
        self.cache_feat_cond = self.model.model.diffusion_model.cache_feat

        model_uncond = self.model.apply_model(x, t, uncond, self.prev_feat_uncond)
        self.cache_feat_uncond = self.model.model.diffusion_model.cache_feat

        model_output = model_uncond + cfg_scale * (model_cond - model_uncond)
    
    if self.model.parameterization == "v":
        e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
    else:
        e_t = model_output

    return e_t


# https://github.com/horseee/DeepCache/blob/master/DeepCache/sd/pipeline_stable_diffusion.py#L86
def sample_from_quad_center(total_numbers, n_samples, center, pow=1.2):
    while pow > 1:
        # Generate linearly spaced values between 0 and a max value
        x_values = np.linspace((-center)**(1/pow), (total_numbers-center)**(1/pow), n_samples+1)
        indices = [0] + [x+center for x in np.unique(np.int32(x_values**pow))[1:-1]]
        if len(indices) == n_samples:
            break
        pow -=0.02
    if pow <= 1:
        raise ValueError("Cannot find suitable pow. Please adjust n_samples or decrease center.")
    return indices, pow


@torch.no_grad()
def hacked_sampler_sample(
    self,
    steps: int,
    shape: Tuple[int],
    cond_img: torch.Tensor,
    positive_prompt: str,
    negative_prompt: str,
    x_T: Optional[torch.Tensor]=None,
    cfg_scale: float=1.,
    cond_fn: Optional[Guidance]=None,
    color_fix_type: str="none"
) -> torch.Tensor:
    self.make_schedule(num_steps=steps)
    
    device = next(self.model.parameters()).device
    b = shape[0]
    if x_T is None:
        img = torch.randn(shape, device=device)
    else:
        img = x_T
    
    time_range = np.flip(self.timesteps) # [1000, 950, 900, ...]
    total_steps = len(self.timesteps)
    iterator = tqdm(time_range, desc="Spaced Sampler", total=total_steps)
    
    allocated = torch.cuda.max_memory_allocated()
    print(f"max allocated VRAM (before condition encoder): {allocated / 1e6:.5f} MB")
    cond = {
        "c_latent": [self.model.apply_condition_encoder(cond_img)],
        "c_crossattn": [self.model.get_learned_conditioning([positive_prompt] * b)]
    }
    uncond = {
        "c_latent": [self.model.apply_condition_encoder(cond_img)],
        "c_crossattn": [self.model.get_learned_conditioning([negative_prompt] * b)]
    }
    allocated = torch.cuda.max_memory_allocated()
    print(f"max allocated VRAM (before denoising process): {allocated / 1e6:.5f} MB")

    use_cache = OptimizationFlag.is_deepcache_enabled()
    if use_cache:
        # https://github.com/horseee/DeepCache/blob/master/DeepCache/sd/pipeline_stable_diffusion.py#L726
        if OptimizationFlag.deepcache.uniform:
            cache_updating_steps = list(range(0, total_steps, OptimizationFlag.deepcache.cache_interval))
        else:
            num_slow_step = total_steps//OptimizationFlag.deepcache.cache_interval
            if total_steps%OptimizationFlag.deepcache.cache_interval != 0:
                num_slow_step += 1
            cache_updating_steps, _ = sample_from_quad_center(
                total_steps, num_slow_step, center=OptimizationFlag.deepcache.center,
                pow=OptimizationFlag.deepcache.pow
            )#[0, 3, 6, 9, 12, 16, 22, 28, 35, 43,]
            print(f"cache_updating_steps: {cache_updating_steps}")
    else:
        cache_updating_steps = list(range(total_steps))
    
    for i, step in enumerate(iterator):
        if i in cache_updating_steps:
            self.prev_feat_cond = None
            self.prev_feat_uncond = None
        else:
            # set cache_feat to the latest features
            self.prev_feat_cond = self.cache_feat_cond
            self.prev_feat_uncond = self.cache_feat_uncond

        ts = torch.full((b,), step, device=device, dtype=torch.long)
        index = torch.full_like(ts, fill_value=total_steps - i - 1)
        img = self.p_sample(
            img, cond, ts, index=index,
            cfg_scale=cfg_scale, uncond=uncond,
            cond_fn=cond_fn
        )
    
    img_pixel = (self.model.decode_first_stage(img) + 1) / 2
    # apply color correction (borrowed from StableSR)
    if color_fix_type == "adain":
        img_pixel = adaptive_instance_normalization(img_pixel, cond_img)
    elif color_fix_type == "wavelet":
        img_pixel = wavelet_reconstruction(img_pixel, cond_img)
    else:
        assert color_fix_type == "none", f"unexpected color fix type: {color_fix_type}"
    return img_pixel


def hacked_basic_transformer_block_init(
    self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True, checkpoint=True,
    disable_self_attn=False
):
    super(self.__class__, self).__init__()
    attn_mode = "softmax-xformers" if OptimizationFlag.is_xformers_enabled() else "softmax"
    assert attn_mode in self.ATTENTION_MODES
    attn_cls = self.ATTENTION_MODES[attn_mode]
    self.disable_self_attn = disable_self_attn
    self.attn1 = attn_cls(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout,
                            context_dim=context_dim if self.disable_self_attn else None)  # is a self-attention if not self.disable_self_attn
    self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
    self.attn2 = attn_cls(query_dim=dim, context_dim=context_dim,
                            heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
    self.norm1 = nn.LayerNorm(dim)
    self.norm2 = nn.LayerNorm(dim)
    self.norm3 = nn.LayerNorm(dim)
    self.checkpoint = checkpoint


def hacked_make_attn(in_channels, attn_type="vanilla", attn_kwargs=None):
    assert attn_type in ["vanilla", "vanilla-xformers", "memory-efficient-cross-attn", "linear", "none"], f'attn_type {attn_type} unknown'
    if OptimizationFlag.is_xformers_enabled() and attn_type == "vanilla":
        attn_type = "vanilla-xformers"
    print(f"making attention of type '{attn_type}' with {in_channels} in_channels")
    if attn_type == "vanilla":
        assert attn_kwargs is None
        return AttnBlock(in_channels)
    elif attn_type == "vanilla-xformers":
        print(f"building MemoryEfficientAttnBlock with {in_channels} in_channels...")
        return MemoryEfficientAttnBlock(in_channels)
    elif type == "memory-efficient-cross-attn":
        attn_kwargs["query_dim"] = in_channels
        return MemoryEfficientCrossAttentionWrapper(**attn_kwargs)
    elif attn_type == "none":
        return nn.Identity(in_channels)
    else:
        raise NotImplementedError()


def hack_everything() -> None:
    ControlLDM.to_fp16 = hacked_cldm_to_fp16
    ControlLDM.to_fp32 = hacked_cldm_to_fp32
    ControlLDM.apply_model = hacked_cldm_apply_model
    ControlledUnetModel.forward = hacked_unet_forward
    ControlNet.forward = hacked_controlnet_forward
    CrossAttention.forward = hacked_attention_forward
    BasicTransformerBlock.__init__ = hacked_basic_transformer_block_init
    model_module.make_attn = hacked_make_attn
    SpacedSampler.sample = hacked_sampler_sample
    SpacedSampler.predict_noise = hacked_sampler_predict_noise
