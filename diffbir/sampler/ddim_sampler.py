from typing import Optional, Tuple, Dict, Literal

import torch
import numpy as np
from tqdm import tqdm

from .sampler import Sampler
from ..model.gaussian_diffusion import extract_into_tensor
from ..model import ControlLDM
from ..utils.common import make_tiled_fn, trace_vram_usage


def make_ddim_timesteps(
    ddim_discr_method: str,
    num_ddim_timesteps: int,
    num_ddpm_timesteps: int,
    verbose: bool = True,
) -> np.ndarray:
    if ddim_discr_method == "uniform":
        c = num_ddpm_timesteps // num_ddim_timesteps
        ddim_timesteps = np.asarray(list(range(0, num_ddpm_timesteps, c)))
    elif ddim_discr_method == "quad":
        ddim_timesteps = (
            (np.linspace(0, np.sqrt(num_ddpm_timesteps * 0.8), num_ddim_timesteps)) ** 2
        ).astype(int)
    else:
        raise NotImplementedError(
            f'There is no ddim discretization method called "{ddim_discr_method}"'
        )

    # assert ddim_timesteps.shape[0] == num_ddim_timesteps
    # add one to get the final alpha values right (the ones from first scale to data during sampling)
    steps_out = ddim_timesteps + 1
    if verbose:
        print(f"Selected timesteps for ddim sampler: {steps_out}")
    return steps_out


def make_ddim_sampling_parameters(
    alphacums: np.ndarray, ddim_timesteps: np.ndarray, eta: float, verbose: bool = True
) -> Tuple[np.ndarray]:
    # select alphas for computing the variance schedule
    alphas = alphacums[ddim_timesteps]
    alphas_prev = np.asarray([alphacums[0]] + alphacums[ddim_timesteps[:-1]].tolist())

    # according the the formula provided in https://arxiv.org/abs/2010.02502
    sigmas = eta * np.sqrt(
        (1 - alphas_prev) / (1 - alphas) * (1 - alphas / alphas_prev)
    )
    if verbose:
        print(
            f"Selected alphas for ddim sampler: a_t: {alphas}; a_(t-1): {alphas_prev}"
        )
        print(
            f"For the chosen value of eta, which is {eta}, "
            f"this results in the following sigma_t schedule for ddim sampler {sigmas}"
        )
    return sigmas, alphas, alphas_prev


class DDIMSampler(Sampler):

    def __init__(
        self,
        betas: np.ndarray,
        parameterization: Literal["eps", "v"],
        rescale_cfg: bool,
        eta: float,
    ) -> "DDIMSampler":
        super().__init__(betas, parameterization, rescale_cfg)
        self.eta = eta

    def make_schedule(
        self,
        ddim_num_steps,
        ddim_discretize="uniform",
    ):
        self.ddim_timesteps = make_ddim_timesteps(
            ddim_discr_method=ddim_discretize,
            num_ddim_timesteps=ddim_num_steps,
            num_ddpm_timesteps=self.num_timesteps,
            verbose=False,
        )
        original_alphas_cumprod = self.training_alphas_cumprod
        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(
            alphacums=original_alphas_cumprod,
            ddim_timesteps=self.ddim_timesteps,
            eta=self.eta,
            verbose=False,
        )
        self.register("ddim_sigmas", ddim_sigmas)
        self.register("ddim_alphas", ddim_alphas)
        self.register("ddim_sqrt_alphas", np.sqrt(ddim_alphas))
        self.register("ddim_alphas_prev", ddim_alphas_prev)
        self.register("ddim_sqrt_one_minus_alphas", np.sqrt(1.0 - ddim_alphas))

    def predict_eps_from_z_and_v(self, x_t, t, v):
        return (
            extract_into_tensor(self.ddim_sqrt_alphas, t, x_t.shape) * v
            + extract_into_tensor(self.ddim_sqrt_one_minus_alphas, t, x_t.shape) * x_t
        )

    @torch.no_grad()
    def p_sample(
        self,
        model: ControlLDM,
        x: torch.Tensor,
        model_t: torch.Tensor,
        t: torch.Tensor,
        cond: Dict[str, torch.Tensor],
        uncond: Optional[Dict[str, torch.Tensor]],
        cfg_scale: float,
    ) -> torch.Tensor:
        if uncond is None or cfg_scale == 1.0:
            model_output = model(x, model_t, cond)
        else:
            x_in = torch.cat([x] * 2)
            t_in = torch.cat([model_t] * 2)
            cond_in = {k: torch.cat([cond[k], uncond[k]]) for k in cond.keys()}
            model_cond, model_uncond = model(x_in, t_in, cond_in).chunk(2)
            model_output = model_uncond + cfg_scale * (model_cond - model_uncond)
        if self.parameterization == "eps":
            e_t = model_output
        else:
            e_t = self.predict_eps_from_z_and_v(x, t, model_output)

        alphas = self.ddim_alphas
        alphas_prev = self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.ddim_sqrt_one_minus_alphas
        sigmas = self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = extract_into_tensor(alphas, t, x.shape)
        a_prev = extract_into_tensor(alphas_prev, t, x.shape)
        sigma_t = extract_into_tensor(sigmas, t, x.shape)
        sqrt_one_minus_at = extract_into_tensor(sqrt_one_minus_alphas, t, x.shape)

        # current prediction for x_0
        pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()

        # direction pointing to x_t
        dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * torch.randn_like(x)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev

    @torch.no_grad()
    def sample(
        self,
        model: ControlLDM,
        device: str,
        steps: int,
        x_size: Tuple[int],
        cond: Dict[str, torch.Tensor],
        uncond: Dict[str, torch.Tensor],
        cfg_scale: float,
        tiled: bool = False,
        tile_size: int = -1,
        tile_stride: int = -1,
        x_T: torch.Tensor | None = None,
        progress: bool = True,
    ) -> torch.Tensor:
        self.make_schedule(ddim_num_steps=steps)
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

        x = x_T
        time_range = np.flip(self.ddim_timesteps)
        total_steps = self.ddim_timesteps.shape[0]
        iterator = tqdm(
            time_range,
            desc="DDIM Sampler",
            total=total_steps,
            disable=not progress,
        )
        bs = x_size[0]

        for i, step in enumerate(iterator):
            model_t = torch.full((bs,), step, device=device, dtype=torch.long)
            t = torch.full((bs,), total_steps - i - 1, device=device, dtype=torch.long)
            cur_cfg_scale = self.get_cfg_scale(cfg_scale, step)
            x = self.p_sample(model, x, model_t, t, cond, uncond, cur_cfg_scale)

        if tiled:
            model.forward = forward
        return x
