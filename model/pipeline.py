from numpy import ndarray
import torch
import numpy as np
import einops
from typing import overload, List, Tuple, Optional, Union, Dict
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.modules import Module
from model.cldm import ControlLDM
from model.cond_fn import Guidance
from model.spaced_sampler import SpacedSampler
from model.bsrnet import RRDBNet
from model.swinir import SwinIR


def pad_iamge_tensors(imgs: torch.Tensor, submultiple: int) -> torch.Tensor:
    h, w = imgs.size(2), imgs.size(3)
    ph = ((h // submultiple) + (h % submultiple != 0)) * submultiple - h
    pw = ((w // submultiple) + (w % submultiple != 0)) * submultiple - w
    return F.pad(imgs, pad=(0, pw, 0, ph), mode="constant", value=0)


def enlarge_image_tensors(imgs: torch.Tensor, min_size: int) -> torch.Tensor:
    h, w = imgs.size(2), imgs.size(3)
    if h == w:
        new_h, new_w = min_size, min_size
    elif h < w:
        new_h, new_w = min_size, int(w * (min_size / h))
    else:
        new_h, new_w = int(h * (min_size / w)), min_size
    return F.interpolate(imgs, size=(new_h, new_w), mode="bicubic")


class Pipeline(object):

    def __init__(self, preprocessor: nn.Module, controller: ControlLDM, cond_fn: Optional[Guidance]) -> None:
        self.preprocessor = preprocessor
        self.controller = controller
        self.cond_fn = cond_fn
        self.device = self.controller.device

    @overload
    def apply_preprocessor(self, images: torch.Tensor) -> torch.Tensor:
        ...

    def apply_controller(
        self,
        cond_imgs: torch.Tensor,
        steps: int,
        strength: float,
        color_fix_type: str,
        tiled: bool,
        tile_size: int,
        tile_stride: int,
        positive_prompt: str,
        negative_prompt: str,
        cfg_scale: float
    ) -> torch.Tensor:
        n, _, ori_h, ori_w = cond_imgs.size()
        
        if min(ori_h, ori_w) < 512:
            cond_imgs = enlarge_image_tensors(cond_imgs, min_size=512)
        r_h, r_w = cond_imgs.size(2), cond_imgs.size(3)
        
        if r_h % 64 == 0 and r_w % 64 == 0:
            padded_cond_imgs = cond_imgs
        else:
            padded_cond_imgs = pad_iamge_tensors(cond_imgs, submultiple=64)
        h, w = padded_cond_imgs.size(2), padded_cond_imgs.size(3)
        
        self.controller.control_scales = [strength] * 13
        latent_shape = (n, 4, h // 8, w // 8)
        sampler = SpacedSampler(self.controller, var_type="fixed_small")
        x_T = torch.randn(latent_shape, device=self.controller.device, dtype=torch.float32)
        if self.cond_fn is not None:
            self.cond_fn.load_target(2 * padded_cond_imgs - 1)

        if not tiled:
            samples = sampler.sample(
                steps=steps, shape=latent_shape, cond_img=padded_cond_imgs,
                positive_prompt=positive_prompt, negative_prompt=negative_prompt, x_T=x_T,
                cfg_scale=cfg_scale, cond_fn=self.cond_fn,
                color_fix_type=color_fix_type
            )
        else:
            samples = sampler.sample_with_mixdiff(
                tile_size=tile_size, tile_stride=tile_stride,
                steps=steps, shape=latent_shape, cond_img=padded_cond_imgs,
                positive_prompt=positive_prompt, negative_prompt=negative_prompt, x_T=x_T,
                cfg_scale=cfg_scale, cond_fn=self.cond_fn,
                color_fix_type=color_fix_type
            )
        
        # remove padding and resize back to condition size
        samples = samples[:, :, :r_h, :r_w]
        samples = F.interpolate(samples, size=(ori_h, ori_w), mode="bicubic", antialias=True)
        
        return samples

    def postprocess_samples(self, samples: torch.Tensor) -> np.ndarray:
        return (
            (einops.rearrange(samples.detach(), "n c h w -> n h w c").cpu().numpy() * 255)
            .clip(0, 255).astype(np.uint8)
        )
    
    def postprocess_cond_imgs(self, cond_imgs: torch.Tensor) -> np.ndarray:
        return (
            (einops.rearrange(cond_imgs.detach(), "n c h w -> n h w c").cpu().numpy() * 255)
            .clip(0, 255).astype(np.uint8)
        )

    def run(
        self,
        images: np.ndarray,
        steps: int,
        strength: float,
        color_fix_type: str,
        tiled: bool,
        tile_size: int,
        tile_stride: int,
        disable_preprocessor: bool,
        positive_prompt: str,
        negative_prompt: str,
        cfg_scale: float
    ) -> Dict[str, np.ndarray]:
        # convery batch lq images from array to tensor
        images = (einops.rearrange(torch.tensor(images), "n h w c -> n c h w") / 255.0).float().to(self.device)
        
        # stage 1: remove degradations from lq iamges
        allocated = torch.cuda.max_memory_allocated()
        print(f"max allocated VRAM (before lq preprocessor): {allocated / 1e6:.5f} MB")
        if disable_preprocessor:
            cond_imgs = images
        else:
            cond_imgs = self.apply_preprocessor(images)
        
        # stage 2: clean images as condition for controller
        samples = self.apply_controller(
            cond_imgs, steps, strength, color_fix_type, tiled, tile_size, tile_stride,
            positive_prompt, negative_prompt, cfg_scale
        )
        
        # return samples and condition images
        return {
            "samples": self.postprocess_samples(samples),
            "cond_imgs": self.postprocess_cond_imgs(cond_imgs)
        }


class BSRPipeline(Pipeline):

    def __init__(self, preprocessor: RRDBNet, controller: ControlLDM, cond_fn: Guidance, upsample_scale: float) -> None:
        super().__init__(preprocessor, controller, cond_fn)
        self.scale = upsample_scale

    def apply_preprocessor(self, images: torch.Tensor) -> torch.Tensor:
        upscaled_images = self.preprocessor(images)
        
        if self.scale != 4:
            # resize to upsample_scale
            h, w = images.size(2), images.size(3)
            final_size = (int(h * self.scale), int(w * self.scale))
            upscaled_images = F.interpolate(upscaled_images, size=final_size, mode="bicubic")
        
        return upscaled_images


class BFRPipeline(Pipeline):

    def __init__(self, preprocessor: SwinIR, controller: ControlLDM, cond_fn: Guidance) -> None:
        super().__init__(preprocessor, controller, cond_fn)
        self.desired_output_size = None

    def apply_preprocessor(self, images: torch.Tensor) -> torch.Tensor:
        ori_h, ori_w = images.size(2), images.size(3)

        # SwinIR is trained on a resolution of 512, so enlarge images to a resolution >= 512
        if min(ori_h, ori_w) < 512:
            images = enlarge_image_tensors(images, min_size=512)
        r_h, r_w = images.size(2), images.size(3)

        # SwinIR requires input size being multiples of 32
        padded_images = pad_iamge_tensors(images, submultiple=32)

        outputs = self.preprocessor(padded_images)[:, :, :r_h, :r_w]
        self.desired_output_size = (ori_h, ori_w)
        
        return outputs
    
    def postprocess_samples(self, samples: Tensor) -> ndarray:
        samples = F.interpolate(samples, size=self.desired_output_size, mode="bicubic", antialias=True)
        return super().postprocess_samples(samples)
    
    def postprocess_cond_imgs(self, cond_imgs: Tensor) -> ndarray:
        cond_imgs = F.interpolate(cond_imgs, size=self.desired_output_size, mode="bicubic", antialias=True)
        return super().postprocess_cond_imgs(cond_imgs)


class BIDPipeline(Pipeline):

    def apply_preprocessor(self, images: torch.Tensor) -> torch.Tensor:
        return self.preprocessor(images)
