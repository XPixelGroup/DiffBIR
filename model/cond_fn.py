from typing import overload
import torch
from torch.nn import functional as F


class Guidance:
    
    def __init__(self, scale, type, t_start, t_stop, space, repeat, loss_type):
        self.scale = scale
        self.type = type
        self.t_start = t_start
        self.t_stop = t_stop
        self.target = None
        self.space = space
        self.repeat = repeat
        self.loss_type = loss_type
    
    def load_target(self, target):
        self.target = target

    def __call__(self, target_x0, pred_x0, t):
        if self.t_stop < t and t < self.t_start:
            # print("sampling with classifier guidance")
            # avoid propagating gradient out of this scope
            pred_x0 = pred_x0.detach().clone()
            target_x0 = target_x0.detach().clone()
            return self.scale * self._forward(target_x0, pred_x0)
        else:
            return None
    
    @overload
    def _forward(self, target_x0, pred_x0): ...


class MSEGuidance(Guidance):
    
    def __init__(self, scale, type, t_start, t_stop, space, repeat, loss_type) -> None:
        super().__init__(
            scale, type, t_start, t_stop, space, repeat, loss_type
        )
    
    @torch.enable_grad()
    def _forward(self, target_x0: torch.Tensor, pred_x0: torch.Tensor):
        # inputs: [-1, 1], nchw, rgb
        pred_x0.requires_grad_(True)
        
        if self.loss_type == "mse":
            loss = (pred_x0 - target_x0).pow(2).mean((1, 2, 3)).sum()
        elif self.loss_type == "downsample_mse":
            # FIXME: scale_factor should be 1/4, not 4
            lr_pred_x0 = F.interpolate(pred_x0, scale_factor=4, mode="bicubic")
            lr_target_x0 = F.interpolate(target_x0, scale_factor=4, mode="bicubic")
            loss = (lr_pred_x0 - lr_target_x0).pow(2).mean((1, 2, 3)).sum()
        else:
            raise ValueError(self.loss_type)
        
        print(f"loss = {loss.item()}")
        return -torch.autograd.grad(loss, pred_x0)[0]
