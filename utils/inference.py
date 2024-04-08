import os
from typing import overload, Generator, Dict
from argparse import Namespace

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf

from model.cldm import ControlLDM
from model.gaussian_diffusion import Diffusion
from model.bsrnet import RRDBNet
from model.scunet import SCUNet
from model.swinir import SwinIR
from utils.common import instantiate_from_config, load_file_from_url, count_vram_usage
from utils.face_restoration_helper import FaceRestoreHelper
from utils.helpers import (
    Pipeline,
    BSRNetPipeline, SwinIRPipeline, SCUNetPipeline,
    bicubic_resize
)
from utils.cond_fn import MSEGuidance, WeightedMSEGuidance


MODELS = {
    ### stage_1 model weights
    "bsrnet": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth",
    # the following checkpoint is up-to-date, but we use the old version in our paper
    # "swinir_face": "https://github.com/zsyOAOA/DifFace/releases/download/V1.0/General_Face_ffhq512.pth",
    "swinir_face": "https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt",
    "scunet_psnr": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
    "swinir_general": "https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt",
    ### stage_2 model weights
    "sd_v21": "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt",
    "v1_face": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_face.pth",
    "v1_general": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_general.pth",
    "v2": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth"
}


def load_model_from_url(url: str) -> Dict[str, torch.Tensor]:
    sd_path = load_file_from_url(url, model_dir="weights")
    sd = torch.load(sd_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if list(sd.keys())[0].startswith("module"):
        sd = {k[len("module."):]: v for k, v in sd.items()}
    return sd


class InferenceLoop:

    def __init__(self, args: Namespace) -> "InferenceLoop":
        self.args = args
        self.loop_ctx = {}
        self.pipeline: Pipeline = None
        self.init_stage1_model()
        self.init_stage2_model()
        self.init_cond_fn()
        self.init_pipeline()

    @overload
    def init_stage1_model(self) -> None:
        ...

    @count_vram_usage
    def init_stage2_model(self) -> None:
        ### load uent, vae, clip
        self.cldm: ControlLDM = instantiate_from_config(OmegaConf.load("configs/inference/cldm.yaml"))
        sd = load_model_from_url(MODELS["sd_v21"])
        unused = self.cldm.load_pretrained_sd(sd)
        print(f"strictly load pretrained sd_v2.1, unused weights: {unused}")
        ### load controlnet
        if self.args.version == "v1":
            if self.args.task == "fr":
                control_sd = load_model_from_url(MODELS["v1_face"])
            elif self.args.task == "sr":
                control_sd = load_model_from_url(MODELS["v1_general"])
            else:
                raise ValueError(f"DiffBIR v1 doesn't support task: {self.args.task}, please use v2 by passsing '--version v2'")
        else:
            control_sd = load_model_from_url(MODELS["v2"])
        self.cldm.load_controlnet_from_ckpt(control_sd)
        print(f"strictly load controlnet weight")
        self.cldm.eval().to(self.args.device)
        ### load diffusion
        self.diffusion: Diffusion = instantiate_from_config(OmegaConf.load("configs/inference/diffusion.yaml"))
        self.diffusion.to(self.args.device)

    def init_cond_fn(self) -> None:
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
            scale=self.args.g_scale, t_start=self.args.g_start, t_stop=self.args.g_stop,
            space=self.args.g_space, repeat=self.args.g_repeat
        )

    @overload
    def init_pipeline(self) -> None:
        ...

    def setup(self) -> None:
        self.output_dir = self.args.output
        os.makedirs(self.output_dir, exist_ok=True)

    def lq_loader(self) -> Generator[np.ndarray, None, None]:
        img_exts = [".png", ".jpg", ".jpeg"]
        if os.path.isdir(self.args.input):
            file_names = sorted([
                file_name for file_name in os.listdir(self.args.input) if os.path.splitext(file_name)[-1] in img_exts
            ])
            file_paths = [os.path.join(self.args.input, file_name) for file_name in file_names]
        else:
            assert os.path.splitext(self.args.input)[-1] in img_exts
            file_paths = [self.args.input]

        def _loader() -> Generator[np.ndarray, None, None]:
            for file_path in file_paths:
                ### load lq
                lq = np.array(Image.open(file_path).convert("RGB"))
                print(f"load lq: {file_path}")
                ### set context for saving results
                self.loop_ctx["file_stem"] = os.path.splitext(os.path.basename(file_path))[0]
                for i in range(self.args.n_samples):
                    self.loop_ctx["repeat_idx"] = i
                    yield lq

        return _loader

    def after_load_lq(self, lq: np.ndarray) -> np.ndarray:
        return lq

    @torch.no_grad()
    def run(self) -> None:
        self.setup()
        # We don't support batch processing since input images may have different size
        loader = self.lq_loader()
        for lq in loader():
            lq = self.after_load_lq(lq)
            sample = self.pipeline.run(
                lq[None], self.args.steps, 1.0, self.args.tiled,
                self.args.tile_size, self.args.tile_stride,
                self.args.pos_prompt, self.args.neg_prompt, self.args.cfg_scale,
                self.args.better_start
            )[0]
            self.save(sample)

    def save(self, sample: np.ndarray) -> None:
        file_stem, repeat_idx = self.loop_ctx["file_stem"], self.loop_ctx["repeat_idx"]
        file_name = f"{file_stem}_{repeat_idx}.png" if self.args.n_samples > 1 else f"{file_stem}.png"
        save_path = os.path.join(self.args.output, file_name)
        Image.fromarray(sample).save(save_path)
        print(f"save result to {save_path}")


class BSRInferenceLoop(InferenceLoop):

    @count_vram_usage
    def init_stage1_model(self) -> None:
        self.bsrnet: RRDBNet = instantiate_from_config(OmegaConf.load("configs/inference/bsrnet.yaml"))
        sd = load_model_from_url(MODELS["bsrnet"])
        self.bsrnet.load_state_dict(sd, strict=True)
        self.bsrnet.eval().to(self.args.device)

    def init_pipeline(self) -> None:
        self.pipeline = BSRNetPipeline(self.bsrnet, self.cldm, self.diffusion, self.cond_fn, self.args.device, self.args.upscale)


class BFRInferenceLoop(InferenceLoop):

    @count_vram_usage
    def init_stage1_model(self) -> None:
        self.swinir_face: SwinIR = instantiate_from_config(OmegaConf.load("configs/inference/swinir.yaml"))
        sd = load_model_from_url(MODELS["swinir_face"])
        self.swinir_face.load_state_dict(sd, strict=True)
        self.swinir_face.eval().to(self.args.device)

    def init_pipeline(self) -> None:
        self.pipeline = SwinIRPipeline(self.swinir_face, self.cldm, self.diffusion, self.cond_fn, self.args.device)

    def after_load_lq(self, lq: np.ndarray) -> np.ndarray:
        # For BFR task, super resolution is achieved by directly upscaling lq
        return bicubic_resize(lq, self.args.upscale)


class BIDInferenceLoop(InferenceLoop):

    @count_vram_usage
    def init_stage1_model(self) -> None:
        self.scunet_psnr: SCUNet = instantiate_from_config(OmegaConf.load("configs/inference/scunet.yaml"))
        sd = load_model_from_url(MODELS["scunet_psnr"])
        self.scunet_psnr.load_state_dict(sd, strict=True)
        self.scunet_psnr.eval().to(self.args.device)

    def init_pipeline(self) -> None:
        self.pipeline = SCUNetPipeline(self.scunet_psnr, self.cldm, self.diffusion, self.cond_fn, self.args.device)

    def after_load_lq(self, lq: np.ndarray) -> np.ndarray:
        # For BID task, super resolution is achieved by directly upscaling lq
        return bicubic_resize(lq, self.args.upscale)


class V1InferenceLoop(InferenceLoop):

    @count_vram_usage
    def init_stage1_model(self) -> None:
        self.swinir: SwinIR = instantiate_from_config(OmegaConf.load("configs/inference/swinir.yaml"))
        if self.args.task == "fr":
            sd = load_model_from_url(MODELS["swinir_face"])
        elif self.args.task == "sr":
            sd = load_model_from_url(MODELS["swinir_general"])
        else:
            raise ValueError(f"DiffBIR v1 doesn't support task: {self.args.task}, please use v2 by passsing '--version v2'")
        self.swinir.load_state_dict(sd, strict=True)
        self.swinir.eval().to(self.args.device)

    def init_pipeline(self) -> None:
        self.pipeline = SwinIRPipeline(self.swinir, self.cldm, self.diffusion, self.cond_fn, self.args.device)

    def after_load_lq(self, lq: np.ndarray) -> np.ndarray:
        # For BFR task, super resolution is achieved by directly upscaling lq
        return bicubic_resize(lq, self.args.upscale)


class UnAlignedBFRInferenceLoop(InferenceLoop):

    @count_vram_usage
    def init_stage1_model(self) -> None:
        self.bsrnet: RRDBNet = instantiate_from_config(OmegaConf.load("configs/inference/bsrnet.yaml"))
        sd = load_model_from_url(MODELS["bsrnet"])
        self.bsrnet.load_state_dict(sd, strict=True)
        self.bsrnet.eval().to(self.args.device)

        self.swinir_face: SwinIR = instantiate_from_config(OmegaConf.load("configs/inference/swinir.yaml"))
        sd = load_model_from_url(MODELS["swinir_face"])
        self.swinir_face.load_state_dict(sd, strict=True)
        self.swinir_face.eval().to(self.args.device)

    def init_pipeline(self) -> None:
        self.pipes = {
            "bg": BSRNetPipeline(self.bsrnet, self.cldm, self.diffusion, self.cond_fn, self.args.device, self.args.upscale),
            "face": SwinIRPipeline(self.swinir_face, self.cldm, self.diffusion, self.cond_fn, self.args.device)
        }
        self.pipeline = self.pipes["face"]

    def setup(self) -> None:
        super().setup()
        self.cropped_face_dir = os.path.join(self.args.output, "cropped_faces")
        os.makedirs(self.cropped_face_dir, exist_ok=True)
        self.restored_face_dir = os.path.join(self.args.output, "restored_faces")
        os.makedirs(self.restored_face_dir, exist_ok=True)
        self.restored_bg_dir = os.path.join(self.args.output, "restored_backgrounds")
        os.makedirs(self.restored_bg_dir, exist_ok=True)

    def lq_loader(self) -> Generator[np.ndarray, None, None]:
        base_loader = super().lq_loader()
        self.face_helper = FaceRestoreHelper(
            device=self.args.device,
            upscale_factor=1,
            face_size=512,
            use_parse=True,
            det_model="retinaface_resnet50"
        )
        
        def _loader() -> Generator[np.ndarray, None, None]:
            for lq in base_loader():
                ### set input image
                self.face_helper.clean_all()
                upscaled_bg = bicubic_resize(lq, self.args.upscale)
                self.face_helper.read_image(upscaled_bg)
                ### get face landmarks for each face
                self.face_helper.get_face_landmarks_5(resize=640, eye_dist_threshold=5)
                self.face_helper.align_warp_face()
                print(f"detect {len(self.face_helper.cropped_faces)} faces")
                ### restore each face (has been upscaeled)
                for i, lq_face in enumerate(self.face_helper.cropped_faces):
                    self.loop_ctx["is_face"] = True
                    self.loop_ctx["face_idx"] = i
                    self.loop_ctx["cropped_face"] = lq_face
                    yield lq_face
                ### restore background (hasn't been upscaled)
                self.loop_ctx["is_face"] = False
                yield lq
        
        return _loader

    def after_load_lq(self, lq: np.ndarray) -> np.ndarray:
        if self.loop_ctx["is_face"]:
            self.pipeline = self.pipes["face"]
        else:
            self.pipeline = self.pipes["bg"]
        return lq

    def save(self, sample: np.ndarray) -> None:
        file_stem, repeat_idx = self.loop_ctx["file_stem"], self.loop_ctx["repeat_idx"]
        if self.loop_ctx["is_face"]:
            face_idx = self.loop_ctx["face_idx"]
            file_name = f"{file_stem}_{repeat_idx}_face_{face_idx}.png"
            Image.fromarray(sample).save(os.path.join(self.restored_face_dir, file_name))

            cropped_face = self.loop_ctx["cropped_face"]
            Image.fromarray(cropped_face).save(os.path.join(self.cropped_face_dir, file_name))

            self.face_helper.add_restored_face(sample)
        else:
            self.face_helper.get_inverse_affine()
            # paste each restored face to the input image
            restored_img = self.face_helper.paste_faces_to_input_image(
                upsample_img=sample
            )
            file_name = f"{file_stem}_{repeat_idx}.png"
            Image.fromarray(sample).save(os.path.join(self.restored_bg_dir, file_name))
            Image.fromarray(restored_img).save(os.path.join(self.output_dir, file_name))
