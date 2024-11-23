import os
from typing import overload, Generator, List, Literal
from argparse import Namespace
from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf
import pandas as pd

from ..utils.common import (
    instantiate_from_config,
    load_model_from_url,
    count_vram_usage,
)
from .pretrained_models import MODELS
from ..pipeline import Pipeline
from ..utils.cond_fn import MSEGuidance, WeightedMSEGuidance
from ..model import ControlLDM, Diffusion
from ..utils.caption import LLaVACaptioner, EmptyCaptioner, LLAVA_AVAILABLE


# @dataclass(frozen=False)
# class LoooArgs:
#     task: Literal["sr", "face", "denoise", "unaligned_face"]
#     upscale: float
#     version: Literal["v1", "v2", "v2.1"]
#     sampler: Literal[
#         "dpm++_m2",
#         "spaced",
#         "ddim",
#         "edm_euler",
#         "edm_euler_a",
#         "edm_heun",
#         "edm_dpm_2",
#         "edm_dpm_2_a",
#         "edm_lms",
#         "edm_dpm++_2s_a",
#         "edm_dpm++_sde",
#         "edm_dpm++_2m",
#         "edm_dpm++_2m_sde",
#         "edm_dpm++_3m_sde",
#     ]
#     steps: int
#     start_point_type: Literal["noise", "cond"]
#     tiled: bool
#     tile_size: int
#     tile_stride: int
#     captioner: Literal["none", "llava"]
#     pos_prompt: str
#     neg_prompt: str
#     cfg_scale: float
#     strength: float
#     rescale_cfg: bool
#     noise_aug: int
#     s_churn: float
#     s_tmin: float
#     s_tmax: float
#     s_noise: float
#     eta: float
#     order: int
#     input: str
#     n_samples: int
#     batch_size: int
#     guidance: bool
#     g_loss: Literal["mse", "w_mse"]
#     g_scale: float
#     output: str
#     device: str


class InferenceLoop:

    def __init__(self, args: Namespace) -> "InferenceLoop":
        self.args = args
        self.loop_ctx = {}
        self.pipeline: Pipeline = None
        self.load_cleaner()
        self.load_cldm()
        self.load_cond_fn()
        self.load_pipeline()
        self.load_captioner()

    @overload
    def load_cleaner(self) -> None: ...

    def load_cldm(self) -> None:
        self.cldm: ControlLDM = instantiate_from_config(
            OmegaConf.load("configs/inference/cldm.yaml")
        )

        # load pre-trained SD weight
        if self.args.version == "v2.1":
            sd_weight = load_model_from_url(MODELS["sd_v2.1_zsnr"])
        else:
            # v1, v2
            sd_weight = load_model_from_url(MODELS["sd_v2.1"])
        unused, missing = self.cldm.load_pretrained_sd(sd_weight)
        print(
            f"load pretrained stable diffusion, "
            f"unused weights: {unused}, missing weights: {missing}"
        )
        # load controlnet weight
        if self.args.version == "v1":
            if self.args.task == "face":
                control_weight = load_model_from_url(MODELS["v1_face"])
            elif self.args.task == "sr" or self.args.task == "denoise":
                control_weight = load_model_from_url(MODELS["v1_general"])
            else:
                raise ValueError(
                    f"DiffBIR v1 doesn't support task: {self.args.task}, "
                    f"please use v2 or v2.1 by passsing '--version'"
                )
        elif self.args.version == "v2":
            control_weight = load_model_from_url(MODELS["v2"])
        else:
            # v2.1
            control_weight = load_model_from_url(MODELS["v2.1"])
        self.cldm.load_controlnet_from_ckpt(control_weight)
        print(f"load controlnet weight")
        self.cldm.eval().to(self.args.device)

        # load diffusion
        if self.args.version in ["v1", "v2"]:
            config = "configs/inference/diffusion.yaml"
        else:
            config = "configs/inference/diffusion_v2.1.yaml"
        self.diffusion: Diffusion = instantiate_from_config(
            OmegaConf.load(config)
        )
        self.diffusion.to(self.args.device)

    def load_cond_fn(self) -> None:
        if not self.args.guidance:
            self.cond_fn = None
            return
        if self.args.g_loss == "mse":
            cond_fn_cls = MSEGuidance
        elif self.args.g_loss == "w_mse":
            cond_fn_cls = WeightedMSEGuidance
        else:
            raise ValueError(self.args.g_loss)
        self.cond_fn = cond_fn_cls(
            self.args.g_scale,
            self.args.g_start,
            self.args.g_stop,
            self.args.g_space,
            self.args.g_repeat,
        )

    @overload
    def load_pipeline(self) -> None: ...

    def load_captioner(self) -> None:
        if self.args.captioner == "none":
            self.captioner = EmptyCaptioner(self.args.device)
        elif self.args.captioner == "llava":
            assert LLAVA_AVAILABLE, "llava is not available in your environment."
            self.captioner = LLaVACaptioner(self.args.device)
        else:
            raise ValueError(f"unsupported captioner: {self.args.captioner}")

    def setup(self) -> None:
        self.save_dir = self.args.output
        os.makedirs(self.save_dir, exist_ok=True)

    def load_lq(self) -> Generator[Image.Image, None, None]:
        img_exts = [".png", ".jpg", ".jpeg"]
        assert os.path.isdir(
            self.args.input
        ), "Please put your low-quality images in a folder."
        for file_name in sorted(os.listdir(self.args.input)):
            stem, ext = os.path.splitext(file_name)
            if ext not in img_exts:
                print(f"{file_name} is not an image, continue")
                continue
            file_path = os.path.join(self.args.input, file_name)
            lq = Image.open(file_path).convert("RGB")
            print(f"load lq: {file_path}")
            self.loop_ctx["file_stem"] = stem
            yield lq

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        return np.array(lq)

    @torch.no_grad()
    def run(self) -> None:
        self.setup()
        for lq in self.load_lq():
            # prepare prompt
            caption = self.captioner(lq)
            pos_prompt = ", ".join(
                [text for text in [caption, self.args.pos_prompt] if text]
            )
            neg_prompt = self.args.neg_prompt
            lq = self.after_load_lq(lq)

            # batch process
            n_samples = self.args.n_samples
            batch_size = self.args.batch_size
            num_batches = (n_samples + batch_size - 1) // batch_size
            samples = []
            for i in range(num_batches):
                n_inputs = min((i + 1) * batch_size, n_samples) - i * batch_size
                batch_samples = self.pipeline.run(
                    np.tile(lq[None], (n_inputs, 1, 1, 1)),
                    self.args.steps,
                    self.args.strength,
                    self.args.tiled,
                    self.args.tile_size,
                    self.args.tile_stride,
                    pos_prompt,
                    neg_prompt,
                    self.args.cfg_scale,
                    self.args.start_point_type,
                    self.args.sampler,
                    self.args.noise_aug,
                    self.args.rescale_cfg,
                    self.args.s_churn,
                    self.args.s_tmin,
                    self.args.s_tmax,
                    self.args.s_noise,
                    self.args.eta,
                    self.args.order,
                )
                samples.extend(list(batch_samples))
            self.save(samples, pos_prompt, neg_prompt)

    def save(self, samples: List[np.ndarray], pos_prompt: str, neg_prompt: str) -> None:
        file_stem = self.loop_ctx["file_stem"]
        assert len(samples) == self.args.n_samples
        for i, sample in enumerate(samples):
            file_name = (
                f"{file_stem}_{i}.png"
                if self.args.n_samples > 1
                else f"{file_stem}.png"
            )
            save_path = os.path.join(self.save_dir, file_name)
            Image.fromarray(sample).save(save_path)
            print(f"save result to {save_path}")
        csv_path = os.path.join(self.save_dir, "prompt.csv")
        df = pd.DataFrame(
            {
                "file_name": [file_stem],
                "pos_prompt": [pos_prompt],
                "neg_prompt": [neg_prompt],
            }
        )
        if os.path.exists(csv_path):
            df.to_csv(csv_path, index=None, mode="a", header=None)
        else:
            df.to_csv(csv_path, index=None)
