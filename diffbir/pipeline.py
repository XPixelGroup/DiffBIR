from typing import overload, Tuple

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from PIL import Image

from .sampler import (
    SpacedSampler,
    DDIMSampler,
    DPMSolverSampler,
    EDMSampler,
)
from .utils.cond_fn import Guidance
from .utils.common import (
    wavelet_reconstruction,
    trace_vram_usage,
    make_tiled_fn,
    VRAMPeakMonitor,
)
from .model import ControlLDM, Diffusion, RRDBNet


def resize_short_edge_to(imgs: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h == w:
        out_h, out_w = size, size
    elif h < w:
        out_h, out_w = size, int(w * (size / h))
    else:
        out_h, out_w = int(h * (size / w)), size

    return F.interpolate(imgs, size=(out_h, out_w), mode="bicubic", antialias=True)


def pad_to_multiples_of(imgs: torch.Tensor, multiple: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h % multiple == 0 and w % multiple == 0:
        return imgs.clone()
    ph, pw = map(lambda x: (x + multiple - 1) // multiple * multiple - x, (h, w))
    return F.pad(imgs, pad=(0, pw, 0, ph), mode="constant", value=0)


class Pipeline:

    def __init__(
        self,
        cleaner: nn.Module,
        cldm: ControlLDM,
        diffusion: Diffusion,
        cond_fn: Guidance | None,
        device: str,
    ) -> None:
        self.cleaner = cleaner
        self.cldm = cldm
        self.diffusion = diffusion
        self.cond_fn = cond_fn
        self.device = device
        self.output_size: Tuple[int, int] = None

    def set_output_size(self, lq_size: Tuple[int]) -> None:
        h, w = lq_size[2:]
        self.output_size = (h, w)

    @overload
    def apply_cleaner(
        self, lq: torch.Tensor, tiled: bool, tile_size: int, tile_stride: int
    ) -> torch.Tensor: ...

    def apply_cldm(
        self,
        cond_img: torch.Tensor,
        steps: int,
        strength: float,
        vae_encoder_tiled: bool,
        vae_encoder_tile_size: int,
        vae_decoder_tiled: bool,
        vae_decoder_tile_size: int,
        cldm_tiled: bool,
        cldm_tile_size: int,
        cldm_tile_stride: int,
        pos_prompt: str,
        neg_prompt: str,
        cfg_scale: float,
        start_point_type: str,
        sampler_type: str,
        noise_aug: int,
        rescale_cfg: bool,
        s_churn: float,
        s_tmin: float,
        s_tmax: float,
        s_noise: float,
        eta: float,
        order: int,
    ) -> torch.Tensor:
        bs, _, h0, w0 = cond_img.shape
        # 1. Pad condition image for VAE encoding (scale factor = 8)
        # 1.1 Whether or not tiled inference is used, the input image size for the VAE must be a multiple of 8.
        if not vae_encoder_tiled and not cldm_tiled:
            # For backward capability, pad condition to be multiples of 64
            cond_img = pad_to_multiples_of(cond_img, multiple=64)
        else:
            cond_img = pad_to_multiples_of(cond_img, multiple=8)
        # 1.2 Check vae encoder tile size
        if vae_encoder_tiled and (
            cond_img.size(2) < vae_encoder_tile_size
            or cond_img.size(3) < vae_encoder_tile_size
        ):
            print("[VAE Encoder]: the input size is tiny and unnecessary to tile.")
            vae_encoder_tiled = False
        # 1.3 If tiled inference is used, then the size of each tile also needs to be a multiple of 8.
        if vae_encoder_tiled:
            if vae_encoder_tile_size % 8 != 0:
                raise ValueError("VAE encoder tile size must be a multiple of 8")
        with VRAMPeakMonitor("encoding condition image"):
            cond = self.cldm.prepare_condition(
                cond_img,
                [pos_prompt] * bs,
                vae_encoder_tiled,
                vae_encoder_tile_size,
            )
            uncond = self.cldm.prepare_condition(
                cond_img,
                [neg_prompt] * bs,
                vae_encoder_tiled,
                vae_encoder_tile_size,
            )
        h1, w1 = cond["c_img"].shape[2:]
        # 2. Pad condition latent for U-Net inference (scale factor = 8)
        # 2.1 Check cldm tile size
        if cldm_tiled and (h1 < cldm_tile_size // 8 or w1 < cldm_tile_size // 8):
            print("[Diffusion]: the input size is tiny and unnecessary to tile.")
            cldm_tiled = False
        # 2.2 Pad conditon latent
        if not cldm_tiled:
            # If tiled inference is not used, apply padding directly.
            cond["c_img"] = pad_to_multiples_of(cond["c_img"], multiple=8)
            uncond["c_img"] = pad_to_multiples_of(uncond["c_img"], multiple=8)
        else:
            # If tiled inference is used, then the latent tile size must be a multiple of 8.
            if cldm_tile_size % 64 != 0:
                raise ValueError("Diffusion tile size must be a multiple of 64")
        h2, w2 = cond["c_img"].shape[2:]
        # 3. Prepare start point of sampling
        if start_point_type == "cond":
            x_0 = cond["c_img"]
            x_T = self.diffusion.q_sample(
                x_0,
                torch.full(
                    (bs,),
                    self.diffusion.num_timesteps - 1,
                    dtype=torch.long,
                    device=self.device,
                ),
                torch.randn(x_0.shape, dtype=torch.float32, device=self.device),
            )
        else:
            x_T = torch.randn((bs, 4, h2, w2), dtype=torch.float32, device=self.device)
        # 4. Noise augmentation
        if noise_aug > 0:
            cond["c_img"] = self.diffusion.q_sample(
                x_start=cond["c_img"],
                t=torch.full(size=(bs,), fill_value=noise_aug, device=self.device),
                noise=torch.randn_like(cond["c_img"]),
            )
            uncond["c_img"] = cond["c_img"].detach().clone()

        if self.cond_fn:
            self.cond_fn.load_target(cond_img * 2 - 1)

        # 5. Set control strength
        control_scales = self.cldm.control_scales
        self.cldm.control_scales = [strength] * 13

        # 6. Run sampler
        betas = self.diffusion.betas
        parameterization = self.diffusion.parameterization
        if sampler_type == "spaced":
            sampler = SpacedSampler(betas, parameterization, rescale_cfg)
        elif sampler_type == "ddim":
            sampler = DDIMSampler(betas, parameterization, rescale_cfg, eta=0)
        elif sampler_type.startswith("dpm"):
            sampler = DPMSolverSampler(
                betas, parameterization, rescale_cfg, sampler_type
            )
        elif sampler_type.startswith("edm"):
            sampler = EDMSampler(
                betas,
                parameterization,
                rescale_cfg,
                sampler_type,
                s_churn,
                s_tmin,
                s_tmax,
                s_noise,
                eta,
                order,
            )
        else:
            raise NotImplementedError(sampler_type)
        with VRAMPeakMonitor("sampling"):
            z = sampler.sample(
                model=self.cldm,
                device=self.device,
                steps=steps,
                x_size=(bs, 4, h2, w2),
                cond=cond,
                uncond=uncond,
                cfg_scale=cfg_scale,
                tiled=cldm_tiled,
                tile_size=cldm_tile_size // 8,
                tile_stride=cldm_tile_stride // 8,
                x_T=x_T,
                progress=True,
            )
            # Remove padding for U-Net input
            z = z[..., :h1, :w1]
        # 7. Decode generated latents
        if vae_decoder_tiled and (
            h1 < vae_decoder_tile_size // 8 or w1 < vae_decoder_tile_size // 8
        ):
            print("[VAE Decoder]: the input size is tiny and unnecessary to tile.")
            vae_decoder_tiled = False
        with VRAMPeakMonitor("decoding generated latent"):
            x = self.cldm.vae_decode(
                z,
                vae_decoder_tiled,
                vae_decoder_tile_size // 8,
            )
        x = x[:, :, :h0, :w0]
        self.cldm.control_scales = control_scales
        return x

    @torch.no_grad()
    def run(
        self,
        lq: np.ndarray,
        steps: int,
        strength: float,
        cleaner_tiled: bool,
        cleaner_tile_size: int,
        cleaner_tile_stride: int,
        vae_encoder_tiled: bool,
        vae_encoder_tile_size: int,
        vae_decoder_tiled: bool,
        vae_decoder_tile_size: int,
        cldm_tiled: bool,
        cldm_tile_size: int,
        cldm_tile_stride: int,
        pos_prompt: str,
        neg_prompt: str,
        cfg_scale: float,
        start_point_type: str,
        sampler_type: str,
        noise_aug: int,
        rescale_cfg: bool,
        s_churn: float,
        s_tmin: float,
        s_tmax: float,
        s_noise: float,
        eta: float,
        order: int,
    ) -> np.ndarray:
        lq_tensor = (
            torch.tensor(lq, dtype=torch.float32, device=self.device)
            .div(255)
            .clamp(0, 1)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        self.set_output_size(lq_tensor.size())
        with VRAMPeakMonitor("applying cleaner"):
            cond_img = self.apply_cleaner(
                lq_tensor, cleaner_tiled, cleaner_tile_size, cleaner_tile_stride
            )
        assert all(x >= 512 for x in cond_img.shape[2:]), (
            "The resolution of stage-1 model output should be greater than 512, "
            "since it will be used as condition for stage-2 model."
        )
        sample = self.apply_cldm(
            cond_img,
            steps,
            strength,
            vae_encoder_tiled,
            vae_encoder_tile_size,
            vae_decoder_tiled,
            vae_decoder_tile_size,
            cldm_tiled,
            cldm_tile_size,
            cldm_tile_stride,
            pos_prompt,
            neg_prompt,
            cfg_scale,
            start_point_type,
            sampler_type,
            noise_aug,
            rescale_cfg,
            s_churn,
            s_tmin,
            s_tmax,
            s_noise,
            eta,
            order,
        )
        sample = F.interpolate(
            wavelet_reconstruction((sample + 1) / 2, cond_img),
            size=self.output_size,
            mode="bicubic",
            antialias=True,
        )
        sample = (
            (sample * 255.0)
            .clamp(0, 255)
            .to(torch.uint8)
            .permute(0, 2, 3, 1)
            .contiguous()
            .cpu()
            .numpy()
        )
        return sample


class BSRNetPipeline(Pipeline):

    def __init__(
        self,
        cleaner: RRDBNet,
        cldm: ControlLDM,
        diffusion: Diffusion,
        cond_fn: Guidance | None,
        device: str,
        upscale: float,
    ) -> None:
        super().__init__(cleaner, cldm, diffusion, cond_fn, device)
        self.upscale = upscale

    def set_output_size(self, lq_size: Tuple[int]) -> None:
        h, w = lq_size[2:]
        self.output_size = (int(h * self.upscale), int(w * self.upscale))

    def apply_cleaner(
        self, lq: torch.Tensor, tiled: bool, tile_size: int, tile_stride: int
    ) -> torch.Tensor:
        if tiled and (lq.size(2) < tile_size or lq.size(3) < tile_size):
            print("[BSRNet]: the input size is tiny and unnecessary to tile.")
            tiled = False

        if tiled:
            model = make_tiled_fn(
                self.cleaner,
                tile_size,
                tile_stride,
                scale_type="up",
                scale=4,
            )
        else:
            model = self.cleaner
        output_upscale4 = model(lq)
        if min(self.output_size) < 512:
            output = resize_short_edge_to(output_upscale4, size=512)
        else:
            output = F.interpolate(
                output_upscale4, size=self.output_size, mode="bicubic", antialias=True
            )
        return output


class SwinIRPipeline(Pipeline):

    def apply_cleaner(
        self, lq: torch.Tensor, tiled: bool, tile_size: int, tile_stride: int
    ) -> torch.Tensor:
        if tiled and (lq.size(2) < tile_size or lq.size(3) < tile_size):
            print("[SwinIR]: the input size is tiny and unnecessary to tile.")
            tiled = False
        if tiled:
            if tile_size % 64 != 0:
                raise ValueError("SwinIR (cleaner) tile size must be a multiple of 64")

        if not tiled:
            # For backward capability, put the resize operation before forward
            if min(lq.shape[2:]) < 512:
                lq = resize_short_edge_to(lq, size=512)
            h0, w0 = lq.shape[2:]
            lq = pad_to_multiples_of(lq, multiple=64)
            output = self.cleaner(lq)[:, :, :h0, :w0]
        else:
            tiled_model = make_tiled_fn(
                self.cleaner,
                size=tile_size,
                stride=tile_stride,
            )
            output = tiled_model(lq)
            if min(output.shape[2:]) < 512:
                output = resize_short_edge_to(output, size=512)
        return output


class SCUNetPipeline(Pipeline):

    def apply_cleaner(
        self, lq: torch.Tensor, tiled: bool, tile_size: int, tile_stride: int
    ) -> torch.Tensor:
        if tiled and (lq.size(2) < tile_size or lq.size(3) < tile_size):
            print("[SCUNet]: the input size is tiny and unnecessary to tile.")
            tiled = False

        if tiled:
            model = make_tiled_fn(
                self.cleaner,
                tile_size,
                tile_stride,
            )
        else:
            model = self.cleaner
        output = model(lq)
        if min(output.shape[2:]) < 512:
            output = resize_short_edge_to(output, size=512)
        return output
