from argparse import Namespace

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import torch

from .loop import InferenceLoop
from ..utils.common import (
    instantiate_from_config,
    VRAMPeakMonitor,
)
from ..pipeline import (
    SwinIRPipeline,
    Pipeline,
)
from ..model import SwinIR, ControlLDM, Diffusion


class CustomInferenceLoop(InferenceLoop):

    def __init__(self, args: Namespace) -> "InferenceLoop":
        self.args = args
        self.train_cfg = OmegaConf.load(args.train_cfg)
        self.loop_ctx = {}
        self.pipeline: Pipeline = None
        with VRAMPeakMonitor("loading cleaner model"):
            self.load_cleaner()
        with VRAMPeakMonitor("loading cldm model"):
            self.load_cldm()
        self.load_cond_fn()
        self.load_pipeline()
        with VRAMPeakMonitor("loading captioner"):
            self.load_captioner()

    def load_cldm(self) -> None:
        self.cldm: ControlLDM = instantiate_from_config(self.train_cfg.model.cldm)

        # load pre-trained SD weight
        sd_weight = torch.load(self.train_cfg.train.sd_path, map_location="cpu")
        sd_weight = sd_weight["state_dict"]
        unused, missing = self.cldm.load_pretrained_sd(sd_weight)
        print(
            f"load pretrained stable diffusion, "
            f"unused weights: {unused}, missing weights: {missing}"
        )
        # load controlnet weight
        control_weight = torch.load(self.args.ckpt, map_location="cpu")
        self.cldm.load_controlnet_from_ckpt(control_weight)
        print(f"load controlnet weight")
        self.cldm.eval().to(self.args.device)
        cast_type = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.args.precision]
        self.cldm.cast_dtype(cast_type)

        # load diffusion
        self.diffusion: Diffusion = instantiate_from_config(
            self.train_cfg.model.diffusion
        )
        self.diffusion.to(self.args.device)

    def load_cleaner(self) -> None:
        # NOTE: Use SwinIR as stage-1 model. Change it if you want.
        self.cleaner: SwinIR = instantiate_from_config(self.train_cfg.model.swinir)
        weight = torch.load(self.train_cfg.train.swinir_path, map_location="cpu")
        if "state_dict" in weight:
            weight = weight["state_dict"]
        weight = {
            (k[len("module.") :] if k.startswith("module.") else k): v
            for k, v in weight.items()
        }
        self.cleaner.load_state_dict(weight, strict=True)
        self.cleaner.eval().to(self.args.device)

    def load_pipeline(self) -> None:
        # NOTE: Choose the correct pipeline if SwinIR is not your stage-1 model.
        self.pipeline = SwinIRPipeline(
            self.cleaner,
            self.cldm,
            self.diffusion,
            self.cond_fn,
            self.args.device,
        )

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        # For SwinIRPipeline, upscaling is achieved by resizing input LQ.
        lq = lq.resize(
            tuple(int(x * self.args.upscale) for x in lq.size), Image.BICUBIC
        )
        return super().after_load_lq(lq)
