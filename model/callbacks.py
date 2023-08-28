from typing import Dict, Any
import os

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.types import STEP_OUTPUT
import torch
import torchvision
from PIL import Image
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only

from .mixins import ImageLoggerMixin


__all__ = [
    "ModelCheckpoint",
    "ImageLogger"
]

class ImageLogger(Callback):
    """
    Log images during training or validating.
    
    TODO: Support validating.
    """
    
    def __init__(
        self,
        log_every_n_steps: int=2000,
        max_images_each_step: int=4,
        log_images_kwargs: Dict[str, Any]=None
    ) -> "ImageLogger":
        super().__init__()
        self.log_every_n_steps = log_every_n_steps
        self.max_images_each_step = max_images_each_step
        self.log_images_kwargs = log_images_kwargs or dict()

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        assert isinstance(pl_module, ImageLoggerMixin)

    @rank_zero_only
    def on_train_batch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs: STEP_OUTPUT,
        batch: Any, batch_idx: int, dataloader_idx: int
    ) -> None:
        if pl_module.global_step % self.log_every_n_steps == 0:
            is_train = pl_module.training
            if is_train:
                pl_module.freeze()
            
            with torch.no_grad():
                # returned images should be: nchw, rgb, [0, 1]
                images: Dict[str, torch.Tensor] = pl_module.log_images(batch, **self.log_images_kwargs)
            
            # save images
            save_dir = os.path.join(pl_module.logger.save_dir, "image_log", "train")
            os.makedirs(save_dir, exist_ok=True)
            for image_key in images:
                image = images[image_key].detach().cpu()
                N = min(self.max_images_each_step, len(image))
                grid = torchvision.utils.make_grid(image[:N], nrow=4)
                # chw -> hwc (hw if gray)
                grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()
                grid = (grid * 255).clip(0, 255).astype(np.uint8)
                filename = "{}_step-{:06}_e-{:06}_b-{:06}.png".format(
                    image_key, pl_module.global_step, pl_module.current_epoch, batch_idx
                )
                path = os.path.join(save_dir, filename)
                Image.fromarray(grid).save(path)
            
            if is_train:
                pl_module.unfreeze()
