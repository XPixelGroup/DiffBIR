from typing import Literal, overload, Dict, Optional, Tuple
import math
import torch
from torch import nn
import numpy as np

from ..model.cldm import ControlLDM


class Sampler(nn.Module):

    def __init__(
        self,
        betas: np.ndarray,
        parameterization: Literal["eps", "v"],
        rescale_cfg: bool,
    ) -> "Sampler":
        super().__init__()
        self.num_timesteps = len(betas)
        self.training_betas = betas
        self.training_alphas_cumprod = np.cumprod(1.0 - betas, axis=0)
        self.context = {}
        self.parameterization = parameterization
        self.rescale_cfg = rescale_cfg

    def register(
        self, name: str, value: np.ndarray, dtype: torch.dtype = torch.float32
    ) -> None:
        self.register_buffer(name, torch.tensor(value, dtype=dtype))

    def get_cfg_scale(self, default_cfg_scale: float, model_t: int) -> float:
        if self.rescale_cfg and default_cfg_scale > 1:
            cfg_scale = 1 + default_cfg_scale * (
                (1 - math.cos(math.pi * ((1000 - model_t) / 1000) ** 5.0)) / 2
            )
        else:
            cfg_scale = default_cfg_scale
        return cfg_scale

    @overload
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
        x_T: Optional[torch.Tensor] = None,
        progress: bool = True,
    ) -> torch.Tensor: ...
