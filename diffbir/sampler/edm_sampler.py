from typing import Literal, Dict, Optional, Callable
import numpy as np
import torch

from .sampler import Sampler
from .k_diffusion import (
    sample_euler,
    sample_euler_ancestral,
    sample_heun,
    sample_dpm_2,
    sample_dpm_2_ancestral,
    sample_lms,
    sample_dpm_fast,
    sample_dpm_adaptive,
    sample_dpmpp_2s_ancestral,
    sample_dpmpp_sde,
    sample_dpmpp_2m,
    sample_dpmpp_2m_sde,
    sample_dpmpp_3m_sde,
    append_dims,
)
from ..model.cldm import ControlLDM
from ..utils.common import make_tiled_fn, trace_vram_usage


class EDMSampler(Sampler):

    TYPE_TO_SOLVER = {
        "euler": (sample_euler, ("s_churn", "s_tmin", "s_tmax", "s_noise")),
        "euler_a": (sample_euler_ancestral, ("eta", "s_noise")),
        "heun": (sample_heun, ("s_churn", "s_tmin", "s_tmax", "s_noise")),
        "dpm_2": (sample_dpm_2, ("s_churn", "s_tmin", "s_tmax", "s_noise")),
        "dpm_2_a": (sample_dpm_2_ancestral, ("eta", "s_noise")),
        "lms": (sample_lms, ("order",)),
        # "dpm_fast": (sample_dpm_fast, ())
        "dpm++_2s_a": (sample_dpmpp_2s_ancestral, ("eta", "s_noise")),
        "dpm++_sde": (sample_dpmpp_sde, ("eta", "s_noise")),
        "dpm++_2m": (sample_dpmpp_2m, ()),
        "dpm++_2m_sde": (sample_dpmpp_2m_sde, ("eta", "s_noise")),
        "dpm++_3m_sde": (sample_dpmpp_3m_sde, ("eta", "s_noise")),
    }

    def __init__(
        self,
        betas: np.ndarray,
        parameterization: Literal["eps", "v"],
        rescale_cfg: bool,
        solver_type: str,
        s_churn: float,
        s_tmin: float,
        s_tmax: float,
        s_noise: float,
        eta: float,
        order: int,
    ) -> "EDMSampler":
        super().__init__(betas, parameterization, rescale_cfg)
        solver_type = solver_type[len("edm_") :]
        solver_fn, solver_hparams = self.TYPE_TO_SOLVER[solver_type]
        params = {
            "s_churn": s_churn,
            "s_tmin": s_tmin,
            "s_tmax": s_tmax,
            "s_noise": s_noise,
            "eta": eta,
            "order": order,
        }

        def wrapped_solver_fn(
            model, x, sigmas, extra_args=None, callback=None, disable=None
        ):
            return solver_fn(
                model=model,
                x=x,
                sigmas=sigmas,
                extra_args=extra_args,
                callback=callback,
                disable=disable,
                **{k: params[k] for k in solver_hparams},
            )

        self.solver_fn = wrapped_solver_fn

    def make_schedule(self, steps: int) -> None:
        timesteps = np.linspace(
            len(self.training_alphas_cumprod) - 1, 0, steps, endpoint=False
        ).astype(int)
        alphas_cumprod = self.training_alphas_cumprod[timesteps].copy()
        # clip alphas cumprod to avoid divide-by-zero
        # alphas_cumprod = np.clip(alphas_cumprod, a_min=1e-6)
        alphas_cumprod[0] = 1e-8
        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        # print(sigmas)
        sigmas = np.append(sigmas, 0)
        timesteps = np.append(timesteps, 0)
        self.register("sigmas", sigmas)
        self.register("timesteps", timesteps, torch.long)

    def convert_to_denoiser(
        self,
        model: ControlLDM,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
    ) -> Callable:
        def denoiser(x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
            if self.parameterization == "eps":
                c_skip = torch.ones_like(sigma)
                c_out = -sigma
                c_in = 1 / (sigma**2 + 1.0) ** 0.5
                c_noise = sigma.clone()
            else:
                c_skip = 1.0 / (sigma**2 + 1.0)
                c_out = -sigma / (sigma**2 + 1.0) ** 0.5
                c_in = 1.0 / (sigma**2 + 1.0) ** 0.5
                c_noise = sigma.clone()
            # convert c_noise to t
            c_noise = self.timesteps[
                (c_noise - self.sigmas[:, None]).abs().argmin(dim=0).view(sigma.shape)
            ]
            # compute current cfg scale
            # all samples in a batch share the same timestep
            cur_cfg_scale = self.get_cfg_scale(cfg_scale, c_noise[0].item())

            c_in, c_out, c_skip = map(
                lambda c: append_dims(c, x.ndim), (c_in, c_out, c_skip)
            )
            if uncond is None or cfg_scale == 1.0:
                model_output = model(x * c_in, c_noise, cond) * c_out + x * c_skip
            else:
                model_cond = model(x * c_in, c_noise, cond) * c_out + x * c_skip
                model_uncond = model(x * c_in, c_noise, uncond) * c_out + x * c_skip
                model_output = model_uncond + cur_cfg_scale * (
                    model_cond - model_uncond
                )
            return model_output

        return denoiser

    @torch.no_grad()
    def sample(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        x_size: torch.Tuple[int],
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        tiled: bool = False,
        tile_size: int = -1,
        tile_stride: int = -1,
        x_T: torch.Tensor | None = None,
        progress: bool = True,
    ) -> torch.Tensor:
        self.make_schedule(steps)
        self.to(device)
        if tiled:
            forward = model.forward
            model.forward = make_tiled_fn(
                lambda x_tile, t, cond, hi, hi_end, wi, wi_end: (
                    forward(
                        x_tile,
                        t,
                        {
                            "c_txt": cond["c_txt"],
                            "c_img": cond["c_img"][..., hi:hi_end, wi:wi_end],
                        },
                    )
                ),
                tile_size,
                tile_stride,
            )
        if x_T is None:
            x_T = torch.randn(x_size, device=device, dtype=torch.float32)

        x = x_T * torch.sqrt(1.0 + self.sigmas[0] ** 2.0)
        denoiser = self.convert_to_denoiser(model, cond, uncond, cfg_scale)
        z = self.solver_fn(
            model=denoiser,
            x=x,
            sigmas=self.sigmas,
            extra_args=None,
            callback=None,
            disable=not progress,
        )
        if tiled:
            model.forward = forward
        return z
