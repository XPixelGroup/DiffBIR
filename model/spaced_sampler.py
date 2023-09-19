from typing import Optional, Tuple, Dict, List, Callable

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_beta_schedule
from model.cond_fn import Guidance
from utils.image import (
    wavelet_reconstruction, adaptive_instance_normalization
)

# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py
def space_timesteps(num_timesteps, section_counts):
    """
    Create a list of timesteps to use from an original diffusion process,
    given the number of timesteps we want to take from equally-sized portions
    of the original process.
    For example, if there's 300 timesteps and the section counts are [10,15,20]
    then the first 100 timesteps are strided to be 10 timesteps, the second 100
    are strided to be 15 timesteps, and the final 100 are strided to be 20.
    If the stride is a string starting with "ddim", then the fixed striding
    from the DDIM paper is used, and only one section is allowed.
    :param num_timesteps: the number of diffusion steps in the original
                          process to divide up.
    :param section_counts: either a list of numbers, or a string containing
                           comma-separated numbers, indicating the step count
                           per section. As a special case, use "ddimN" where N
                           is a number of steps to use the striding from the
                           DDIM paper.
    :return: a set of diffusion steps from the original process to use.
    """
    if isinstance(section_counts, str):
        if section_counts.startswith("ddim"):
            desired_count = int(section_counts[len("ddim") :])
            for i in range(1, num_timesteps):
                if len(range(0, num_timesteps, i)) == desired_count:
                    return set(range(0, num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {num_timesteps} steps with an integer stride"
            )
        section_counts = [int(x) for x in section_counts.split(",")]
    size_per = num_timesteps // len(section_counts)
    extra = num_timesteps % len(section_counts)
    start_idx = 0
    all_steps = []
    for i, section_count in enumerate(section_counts):
        size = size_per + (1 if i < extra else 0)
        if size < section_count:
            raise ValueError(
                f"cannot divide section of {size} steps into {section_count}"
            )
        if section_count <= 1:
            frac_stride = 1
        else:
            frac_stride = (size - 1) / (section_count - 1)
        cur_idx = 0.0
        taken_steps = []
        for _ in range(section_count):
            taken_steps.append(start_idx + round(cur_idx))
            cur_idx += frac_stride
        all_steps += taken_steps
        start_idx += size
    return set(all_steps)


# https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/gaussian_diffusion.py
def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.
    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    try:
        # float64 as default. float64 is not supported by mps device.
        res = torch.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    except:
        # to be compatible with mps
        res = torch.from_numpy(arr.astype(np.float32)).to(device=timesteps.device)[timesteps].float()
        
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


class SpacedSampler:
    """
    Implementation for spaced sampling schedule proposed in IDDPM. This class is designed
    for sampling ControlLDM.
    
    https://arxiv.org/pdf/2102.09672.pdf
    """
    
    def __init__(
        self,
        model: "ControlLDM",
        schedule: str="linear",
        var_type: str="fixed_small"
    ) -> "SpacedSampler":
        self.model = model
        self.original_num_steps = model.num_timesteps
        self.schedule = schedule
        self.var_type = var_type

    def make_schedule(self, num_steps: int) -> None:
        """
        Initialize sampling parameters according to `num_steps`.
        
        Args:
            num_steps (int): Sampling steps.

        Returns:
            None
        """
        # NOTE: this schedule, which generates betas linearly in log space, is a little different
        # from guided diffusion.
        original_betas = make_beta_schedule(
            self.schedule, self.original_num_steps, linear_start=self.model.linear_start,
            linear_end=self.model.linear_end
        )
        original_alphas = 1.0 - original_betas
        original_alphas_cumprod = np.cumprod(original_alphas, axis=0)
        
        # calcualte betas for spaced sampling
        # https://github.com/openai/guided-diffusion/blob/main/guided_diffusion/respace.py
        used_timesteps = space_timesteps(self.original_num_steps, str(num_steps))
        print(f"timesteps used in spaced sampler: \n\t{sorted(list(used_timesteps))}")
        
        betas = []
        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(original_alphas_cumprod):
            if i in used_timesteps:
                # marginal distribution is the same as q(x_{S_t}|x_0)
                betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        assert len(betas) == num_steps
        betas = np.array(betas, dtype=np.float64)
        self.betas = betas
        
        self.timesteps = np.array(sorted(list(used_timesteps)), dtype=np.int32) # e.g. [0, 10, 20, ...]
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (num_steps, )
        
        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor]=None
    ) -> torch.Tensor:
        """
        Implement the marginal distribution q(x_t|x_0).

        Args:
            x_start (torch.Tensor): Images (NCHW) sampled from data distribution.
            t (torch.Tensor): Timestep (N) for diffusion process. `t` serves as an index
                to get parameters for each timestep.
            noise (torch.Tensor, optional): Specify the noise (NCHW) added to `x_start`.

        Returns:
            x_t (torch.Tensor): The noisy images.
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            _extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + _extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(
        self,
        x_start: torch.Tensor,
        x_t: torch.Tensor,
        t: torch.Tensor
    ) -> Tuple[torch.Tensor]:
        """
        Implement the posterior distribution q(x_{t-1}|x_t, x_0).
        
        Args:
            x_start (torch.Tensor): The predicted images (NCHW) in timestep `t`.
            x_t (torch.Tensor): The sampled intermediate variables (NCHW) of timestep `t`.
            t (torch.Tensor): Timestep (N) of `x_t`. `t` serves as an index to get 
                parameters for each timestep.
        
        Returns:
            posterior_mean (torch.Tensor): Mean of the posterior distribution.
            posterior_variance (torch.Tensor): Variance of the posterior distribution.
            posterior_log_variance_clipped (torch.Tensor): Log variance of the posterior distribution.
        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            _extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + _extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = _extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = _extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def _predict_xstart_from_eps(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        eps: torch.Tensor
    ) -> torch.Tensor:
        assert x_t.shape == eps.shape
        return (
            _extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - _extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )
    
    def predict_noise(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        cfg_scale: float,
        uncond: Optional[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        if uncond is None or cfg_scale == 1.:
            model_output = self.model.apply_model(x, t, cond)
        else:
            # apply classifier-free guidance
            model_cond = self.model.apply_model(x, t, cond)
            model_uncond = self.model.apply_model(x, t, uncond)
            model_output = model_uncond + cfg_scale * (model_cond - model_uncond)
        
        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        return e_t
    
    def apply_cond_fn(
        self,
        x: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        t: torch.Tensor,
        index: torch.Tensor,
        cond_fn: Guidance,
        cfg_scale: float,
        uncond: Optional[Dict[str, torch.Tensor]]
    ) -> torch.Tensor:
        device = x.device
        t_now = int(t[0].item()) + 1
        # ----------------- predict noise and x0 ----------------- #
        e_t = self.predict_noise(
            x, t, cond, cfg_scale, uncond
        )
        pred_x0: torch.Tensor = self._predict_xstart_from_eps(x_t=x, t=index, eps=e_t)
        model_mean, _, _ = self.q_posterior_mean_variance(
            x_start=pred_x0, x_t=x, t=index
        )
        
        # apply classifier guidance for multiple times
        for _ in range(cond_fn.repeat):
            # ----------------- compute gradient for x0 in latent space ----------------- #
            target, pred = None, None
            if cond_fn.space == "latent":
                target = self.model.get_first_stage_encoding(
                    self.model.encode_first_stage(cond_fn.target.to(device))
                )
                pred = pred_x0
            elif cond_fn.space == "rgb":
                # We need to backward gradient to x0 in latent space, so it's required
                # to trace the computation graph while decoding the latent.
                with torch.enable_grad():
                    pred_x0.requires_grad_(True)
                    target = cond_fn.target.to(device)
                    pred = self.model.decode_first_stage_with_grad(pred_x0)
            else:
                raise NotImplementedError(cond_fn.space)
            delta_pred = cond_fn(target, pred, t_now)
            
            # ----------------- apply classifier guidance ----------------- #
            if delta_pred is not None:
                if cond_fn.space == "rgb":
                    # compute gradient for pred_x0
                    pred.backward(delta_pred)
                    delta_pred_x0 = pred_x0.grad
                    # update prex_x0
                    pred_x0 += delta_pred_x0
                    # our classifier guidance is equivalent to multiply delta_pred_x0
                    # by a constant and then add it to model_mean, We set the constant
                    # to 0.5
                    model_mean += 0.5 * delta_pred_x0
                    pred_x0.grad.zero_()
                else:
                    delta_pred_x0 = delta_pred
                    pred_x0 += delta_pred_x0
                    model_mean += 0.5 * delta_pred_x0
            else:
                # means stop guidance
                break
        
        return model_mean.detach().clone(), pred_x0.detach().clone()
    
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        t: torch.Tensor,
        index: torch.Tensor,
        cfg_scale: float,
        uncond: Optional[Dict[str, torch.Tensor]],
        cond_fn: Optional[Guidance]
    ) -> torch.Tensor:
        # variance of posterior distribution q(x_{t-1}|x_t, x_0)
        model_variance = {
            "fixed_large": np.append(self.posterior_variance[1], self.betas[1:]),
            "fixed_small": self.posterior_variance
        }[self.var_type]
        model_variance = _extract_into_tensor(model_variance, index, x.shape)
        
        # mean of posterior distribution q(x_{t-1}|x_t, x_0)
        if cond_fn is not None:
            # apply classifier guidance
            model_mean, pred_x0 = self.apply_cond_fn(
                x, cond, t, index, cond_fn,
                cfg_scale, uncond
            )
        else:
            e_t = self.predict_noise(
                x, t, cond, cfg_scale, uncond
            )
            pred_x0 = self._predict_xstart_from_eps(x_t=x, t=index, eps=e_t)
            model_mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_x0, x_t=x, t=index
            )
        
        # sample x_t from q(x_{t-1}|x_t, x_0)
        noise = torch.randn_like(x)
        nonzero_mask = (
            (index != 0).float().view(-1, *([1] * (len(x.shape) - 1)))
        )
        x_prev = model_mean + nonzero_mask * torch.sqrt(model_variance) * noise
        return x_prev
    
    @torch.no_grad()
    def sample_with_mixdiff(
        self,
        tile_size: int,
        tile_stride: int,
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
        def _sliding_windows(h: int, w: int, tile_size: int, tile_stride: int) -> Tuple[int, int, int, int]:
            hi_list = list(range(0, h - tile_size + 1, tile_stride))
            if (h - tile_size) % tile_stride != 0:
                hi_list.append(h - tile_size)
            
            wi_list = list(range(0, w - tile_size + 1, tile_stride))
            if (w - tile_size) % tile_stride != 0:
                wi_list.append(w - tile_size)
            
            coords = []
            for hi in hi_list:
                for wi in wi_list:
                    coords.append((hi, hi + tile_size, wi, wi + tile_size))
            return coords
        
        # make sampling parameters (e.g. sigmas)
        self.make_schedule(num_steps=steps)
        
        device = next(self.model.parameters()).device
        b, _, h, w = shape
        if x_T is None:
            img = torch.randn(shape, dtype=torch.float32, device=device)
        else:
            img = x_T
        # create buffers for accumulating predicted noise of different diffusion process
        noise_buffer = torch.zeros_like(img)
        count = torch.zeros(shape, dtype=torch.long, device=device)
        # timesteps iterator
        time_range = np.flip(self.timesteps) # [1000, 950, 900, ...]
        total_steps = len(self.timesteps)
        iterator = tqdm(time_range, desc="Spaced Sampler", total=total_steps)
        
        # sampling loop
        for i, step in enumerate(iterator):
            ts = torch.full((b,), step, device=device, dtype=torch.long)
            index = torch.full_like(ts, fill_value=total_steps - i - 1)
            
            # predict noise for each tile
            tiles_iterator = tqdm(_sliding_windows(h, w, tile_size // 8, tile_stride // 8))
            for hi, hi_end, wi, wi_end in tiles_iterator:
                tiles_iterator.set_description(f"Process tile with location ({hi} {hi_end}) ({wi} {wi_end})")
                # noisy latent of this diffusion process (tile) at this step
                tile_img = img[:, :, hi:hi_end, wi:wi_end]
                # prepare condition for this tile
                tile_cond_img = cond_img[:, :, hi * 8:hi_end * 8, wi * 8: wi_end * 8]
                tile_cond = {
                    "c_latent": [self.model.apply_condition_encoder(tile_cond_img)],
                    "c_crossattn": [self.model.get_learned_conditioning([positive_prompt] * b)]
                }
                tile_uncond = {
                    "c_latent": [self.model.apply_condition_encoder(tile_cond_img)],
                    "c_crossattn": [self.model.get_learned_conditioning([negative_prompt] * b)]
                }
                # TODO: tile_cond_fn
                
                # predict noise for this tile
                tile_noise = self.predict_noise(tile_img, ts, tile_cond, cfg_scale, tile_uncond)
                
                # accumulate noise
                noise_buffer[:, :, hi:hi_end, wi:wi_end] += tile_noise
                count[:, :, hi:hi_end, wi:wi_end] += 1
            
            # average on noise (score)
            noise_buffer.div_(count)
            # sample previous latent
            pred_x0 = self._predict_xstart_from_eps(x_t=img, t=index, eps=noise_buffer)
            mean, _, _ = self.q_posterior_mean_variance(
                x_start=pred_x0, x_t=img, t=index
            )
            variance = {
                "fixed_large": np.append(self.posterior_variance[1], self.betas[1:]),
                "fixed_small": self.posterior_variance
            }[self.var_type]
            variance = _extract_into_tensor(variance, index, noise_buffer.shape)
            
            nonzero_mask = (
                (index != 0).float().view(-1, *([1] * (len(noise_buffer.shape) - 1)))
            )
            img = mean + nonzero_mask * torch.sqrt(variance) * torch.randn_like(mean)
            
            noise_buffer.zero_()
            count.zero_()
        
        # decode samples of each diffusion process
        img_buffer = torch.zeros_like(cond_img)
        count = torch.zeros_like(cond_img, dtype=torch.long)
        for hi, hi_end, wi, wi_end in _sliding_windows(h, w, tile_size // 8, tile_stride // 8):
            tile_img = img[:, :, hi:hi_end, wi:wi_end]
            tile_img_pixel = (self.model.decode_first_stage(tile_img) + 1) / 2
            tile_cond_img = cond_img[:, :, hi * 8:hi_end * 8, wi * 8: wi_end * 8]
            # apply color correction (borrowed from StableSR)
            if color_fix_type == "adain":
                tile_img_pixel = adaptive_instance_normalization(tile_img_pixel, tile_cond_img)
            elif color_fix_type == "wavelet":
                tile_img_pixel = wavelet_reconstruction(tile_img_pixel, tile_cond_img)
            else:
                assert color_fix_type == "none", f"unexpected color fix type: {color_fix_type}"
            img_buffer[:, :, hi * 8:hi_end * 8, wi * 8: wi_end * 8] += tile_img_pixel
            count[:, :, hi * 8:hi_end * 8, wi * 8: wi_end * 8] += 1
        img_buffer.div_(count)
        
        return img_buffer
    
    @torch.no_grad()
    def sample(
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
        
        cond = {
            "c_latent": [self.model.apply_condition_encoder(cond_img)],
            "c_crossattn": [self.model.get_learned_conditioning([positive_prompt] * b)]
        }
        uncond = {
            "c_latent": [self.model.apply_condition_encoder(cond_img)],
            "c_crossattn": [self.model.get_learned_conditioning([negative_prompt] * b)]
        }
        for i, step in enumerate(iterator):
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
