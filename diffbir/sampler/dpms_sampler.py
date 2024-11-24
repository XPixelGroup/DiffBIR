from typing import Tuple, Dict, Literal

import torch
import numpy as np

from .sampler import Sampler
from .dpm_solver_pytorch import (
    NoiseScheduleVP,
    model_wrapper,
    DPM_Solver,
)
from ..utils.cond_fn import Guidance
from ..model.cldm import ControlLDM
from ..utils.common import make_tiled_fn, trace_vram_usage


class DPMSolverSampler(Sampler):

    def __init__(
        self,
        betas: np.ndarray,
        parameterization: Literal["eps", "v"],
        rescale_cfg: bool,
        model_spec: str,
    ) -> "DPMSolverSampler":
        super().__init__(betas, parameterization, rescale_cfg)
        if parameterization == "eps":
            self.model_type = "noise"
        elif parameterization == "v":
            self.model_type = "v"
        else:
            raise ValueError(parameterization)
        # parse samping args from string
        # e.g. dpm++_s2 => solver_type=dpmsolver++, method=singlestep, order=2
        solver_type, (method, order) = model_spec.split("_")
        self.solver_type = {"dpm": "dpmsolver", "dpm++": "dpmsolver++"}[solver_type]
        self.method = {"s": "singlestep", "m": "multistep"}[method]
        self.order = {"1": 1, "2": 2, "3": 3}[order]
        self.register("betas", betas)

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

        noise_schedule = NoiseScheduleVP(schedule="discrete", betas=self.betas)
        model_fn = model_wrapper(
            lambda x, t, c: model(x, t, c),
            noise_schedule,
            model_type=self.model_type,
            guidance_type="classifier-free",
            condition=cond,
            unconditional_condition=uncond,
            guidance_scale=cfg_scale,
            cfg_rescale=self.rescale_cfg,
        )
        dpm_solver = DPM_Solver(
            model_fn, noise_schedule, algorithm_type=self.solver_type
        )
        x = dpm_solver.sample(
            x_T,
            steps=steps,
            skip_type="time_uniform",
            method=self.method,
            order=self.order,
            return_intermediate=False,
        )
        if tiled:
            model.forward = forward
        return x
