from typing import Tuple, Set, List, Dict

import torch
from torch import nn

from model import (
    ControlledUnetModel, ControlNet,
    AutoencoderKL, FrozenOpenCLIPEmbedder
)
from utils.common import sliding_windows, count_vram_usage, gaussian_weights


def disabled_train(self: nn.Module) -> nn.Module:
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


class ControlLDM(nn.Module):

    def __init__(
        self,
        unet_cfg,
        vae_cfg,
        clip_cfg,
        controlnet_cfg,
        latent_scale_factor
    ):
        super().__init__()
        self.unet = ControlledUnetModel(**unet_cfg)
        self.vae = AutoencoderKL(**vae_cfg)
        self.clip = FrozenOpenCLIPEmbedder(**clip_cfg)
        self.controlnet = ControlNet(**controlnet_cfg)
        self.scale_factor = latent_scale_factor
        self.control_scales = [1.0] * 13

    @torch.no_grad()
    def load_pretrained_sd(self, sd: Dict[str, torch.Tensor]) -> Set[str]:
        module_map = {
            "unet": "model.diffusion_model",
            "vae": "first_stage_model",
            "clip": "cond_stage_model",
        }
        modules = [("unet", self.unet), ("vae", self.vae), ("clip", self.clip)]
        used = set()
        for name, module in modules:
            init_sd = {}
            scratch_sd = module.state_dict()
            for key in scratch_sd:
                target_key = ".".join([module_map[name], key])
                init_sd[key] = sd[target_key].clone()
                used.add(target_key)
            module.load_state_dict(init_sd, strict=True)
        unused = set(sd.keys()) - used
        # NOTE: this is slightly different from previous version, which haven't switched
        # the UNet to eval mode and disabled the requires_grad flag.
        for module in [self.vae, self.clip, self.unet]:
            module.eval()
            module.train = disabled_train
            for p in module.parameters():
                p.requires_grad = False
        return unused
    
    @torch.no_grad()
    def load_controlnet_from_ckpt(self, sd: Dict[str, torch.Tensor]) -> None:
        self.controlnet.load_state_dict(sd, strict=True)

    @torch.no_grad()
    def load_controlnet_from_unet(self) -> Tuple[Set[str]]:
        unet_sd = self.unet.state_dict()
        scratch_sd = self.controlnet.state_dict()
        init_sd = {}
        init_with_new_zero = set()
        init_with_scratch = set()
        for key in scratch_sd:
            if key in unet_sd:
                this, target = scratch_sd[key], unet_sd[key]
                if this.size() == target.size():
                    init_sd[key] = target.clone()
                else:
                    d_ic = this.size(1) - target.size(1)
                    oc, _, h, w = this.size()
                    zeros = torch.zeros((oc, d_ic, h, w), dtype=target.dtype)
                    init_sd[key] = torch.cat((target, zeros), dim=1)
                    init_with_new_zero.add(key)
            else:
                init_sd[key] = scratch_sd[key].clone()
                init_with_scratch.add(key)
        self.controlnet.load_state_dict(init_sd, strict=True)
        return init_with_new_zero, init_with_scratch
    
    def vae_encode(self, image: torch.Tensor, sample: bool=True) -> torch.Tensor:
        if sample:
            return self.vae.encode(image).sample() * self.scale_factor
        else:
            return self.vae.encode(image).mode() * self.scale_factor
    
    def vae_encode_tiled(self, image: torch.Tensor, tile_size: int, tile_stride: int, sample: bool=True) -> torch.Tensor:
        bs, _, h, w = image.shape
        z = torch.zeros((bs, 4, h // 8, w // 8), dtype=torch.float32, device=image.device)
        count = torch.zeros_like(z, dtype=torch.float32)
        weights = gaussian_weights(tile_size // 8, tile_size // 8)[None, None]
        weights = torch.tensor(weights, dtype=torch.float32, device=image.device)
        tiles = sliding_windows(h // 8, w // 8, tile_size // 8, tile_stride // 8)
        for hi, hi_end, wi, wi_end in tiles:
            tile_image = image[:, :, hi * 8:hi_end * 8, wi * 8:wi_end * 8]
            z[:, :, hi:hi_end, wi:wi_end] += self.vae_encode(tile_image, sample=sample) * weights
            count[:, :, hi:hi_end, wi:wi_end] += weights
        z.div_(count)
        return z
    
    def vae_decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(z / self.scale_factor)
    
    @count_vram_usage
    def vae_decode_tiled(self, z: torch.Tensor, tile_size: int, tile_stride: int) -> torch.Tensor:
        bs, _, h, w = z.shape
        image = torch.zeros((bs, 3, h * 8, w * 8), dtype=torch.float32, device=z.device)
        count = torch.zeros_like(image, dtype=torch.float32)
        weights = gaussian_weights(tile_size * 8, tile_size * 8)[None, None]
        weights = torch.tensor(weights, dtype=torch.float32, device=z.device)
        tiles = sliding_windows(h, w, tile_size, tile_stride)
        for hi, hi_end, wi, wi_end in tiles:
            tile_z = z[:, :, hi:hi_end, wi:wi_end]
            image[:, :, hi * 8:hi_end * 8, wi * 8:wi_end * 8] += self.vae_decode(tile_z) * weights
            count[:, :, hi * 8:hi_end * 8, wi * 8:wi_end * 8] += weights
        image.div_(count)
        return image

    def prepare_condition(self, clean: torch.Tensor, txt: List[str]) -> Dict[str, torch.Tensor]:
        return dict(
            c_txt=self.clip.encode(txt),
            c_img=self.vae_encode(clean * 2 - 1, sample=False)
        )
    
    @count_vram_usage
    def prepare_condition_tiled(self, clean: torch.Tensor, txt: List[str], tile_size: int, tile_stride: int) -> Dict[str, torch.Tensor]:
        return dict(
            c_txt=self.clip.encode(txt),
            c_img=self.vae_encode_tiled(clean * 2 - 1, tile_size, tile_stride, sample=False)
        )

    def forward(self, x_noisy, t, cond):
        c_txt = cond["c_txt"]
        c_img = cond["c_img"]
        control = self.controlnet(
            x=x_noisy, hint=c_img,
            timesteps=t, context=c_txt
        )
        control = [c * scale for c, scale in zip(control, self.control_scales)]
        eps = self.unet(
            x=x_noisy, timesteps=t,
            context=c_txt, control=control, only_mid_control=False
        )
        return eps
