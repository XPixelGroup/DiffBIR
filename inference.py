import os
from typing import Optional, overload, Generator
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf
from torch import nn

from ldm.xformers_state import disable_xformers
from model.cond_fn import Guidance, MSEGuidance
from utils.common import instantiate_from_config, load_state_dict
from model.pipeline import BSRPipeline, BFRPipeline, BIDPipeline
from utils.file import load_file_from_url
from utils.face_restoration_helper import FaceRestoreHelper


def check_device(device: str) -> str:
    if device == "cuda":
        # check if CUDA is available
        if not torch.cuda.is_available():
            print(
                "CUDA not available because the current PyTorch install was not "
                "built with CUDA enabled."
            )
            device = "cpu"
    else:
        # xformers only support CUDA. Disable xformers when using cpu or mps.
        disable_xformers()
        if device == "mps":
            # check if MPS is available
            if not torch.backends.mps.is_available():
                if not torch.backends.mps.is_built():
                    print(
                        "MPS not available because the current PyTorch install was not "
                        "built with MPS enabled."
                    )
                    device = "cpu"
                else:
                    print(
                        "MPS not available because the current MacOS version is not 12.3+ "
                        "and/or you do not have an MPS-enabled device on this machine."
                    )
                    device = "cpu"
    print(f"using device {device}")
    return device


def load_model(config_path: str, ckpt_path: str, ckpt_url: Optional[str]=None, save_dir: str="weights") -> nn.Module:
    print(f"load model config file and instantiate: {config_path}")
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config)
    if not os.path.exists(ckpt_path):
        print(f"download {ckpt_path}")
        ckpt_path = load_file_from_url(ckpt_url, save_dir)
    print(f"load state dict from {ckpt_path}")
    load_state_dict(model, torch.load(ckpt_path), strict=True)
    return model


def build_cond_fn(args: Namespace) -> Optional[Guidance]:
    if not args.use_guidance:
        return None
    if args.g_loss == "mse":
        return MSEGuidance(
            scale=args.g_scale, t_start=args.g_t_start, t_stop=args.g_t_stop,
            space=args.g_space, repeat=args.g_repeat
        )
    else:
        raise ValueError(args.g_loss)


def image_resize(image: np.ndarray, scale: float) -> np.ndarray:
    return np.array(
        Image.fromarray(image).resize(tuple(int(x * scale) for x in image.shape[1::-1]), Image.BICUBIC)
    )


class InferenceLoop:

    def __init__(self, args: Namespace) -> "InferenceLoop":
        self.args = args
        self.build_pipeline()
        self.loop_context = {}
    
    @overload
    def build_pipeline(self) -> None:
        ...

    def get_lq_generator(self) -> Generator:
        img_exts = [".png", ".jpg", ".jpeg"]
        file_names = sorted([
            file_name for file_name in os.listdir(self.args.input) if os.path.splitext(file_name)[-1] in img_exts
        ])
        
        def _generator() -> np.ndarray:
            for file_name in file_names:
                # save current file name for `save_iamge`
                self.loop_context["file_stem"] = os.path.splitext(file_name)[0]
                file_path = os.path.join(self.args.input, file_name)

                image = np.array(Image.open(file_path).convert("RGB"))
                for i in range(self.args.n_samples):
                    # save repeat index for `save_iamge`
                    self.loop_context["repeat_idx"] = i
                    yield image
        
        return _generator
    
    @overload
    def restore_image(self, image: np.ndarray) -> np.ndarray:
        ...

    def save_image(self, sample: np.ndarray) -> None:
        file_stem, repeat_idx = self.loop_context["file_stem"], self.loop_context["repeat_idx"]
        file_name = f"{file_stem}_{repeat_idx}.png" if self.args.n_samples >= 1 else f"{file_stem}.png"
        file_path = os.path.join(self.args.output, file_name)
        Image.fromarray(sample).save(file_path)
        print(f"save result to {file_path}")

    def setup(self) -> None:
        self.output_dir = self.args.output
        os.makedirs(self.output_dir, exist_ok=True)

    @torch.no_grad()
    def run(self) -> None:
        self.setup()
        # process image one by one, batch processing is not necessary
        generator = self.get_lq_generator()
        for image in generator():
            if self.args.autocast:
                print("enable autocast")
            with torch.autocast(device_type="cuda", enabled=self.args.autocast):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                sample = self.restore_image(image)
                end.record()
                torch.cuda.synchronize()
                print(f"time cost: {(start.elapsed_time(end)) / 1000:.5f} seconds")
                allocated = torch.cuda.max_memory_allocated()
                print(f"max allocated VRAM: {allocated / 1e6:.5f} MB")
                self.save_image(sample)


class BSRInferenceLoop(InferenceLoop):

    def build_pipeline(self) -> None:
        bsrnet = load_model("configs/model/bsrnet.yaml", "weights/bsrnet.ckpt").to(self.args.device).eval()
        controller = load_model("configs/model/cldm.yaml", "weights/controller.ckpt").to(self.args.device).eval()
        cond_fn = build_cond_fn(self.args)
        self.pipeline = BSRPipeline(bsrnet, controller, cond_fn, upsample_scale=self.args.upsample_scale)

    def restore_image(self, image: np.ndarray) -> np.ndarray:
        output = self.pipeline.run(
            image[None], steps=self.args.steps, strength=1.0, color_fix_type=self.args.color_fix_type,
            tiled=self.args.tiled, tile_size=self.args.tile_size, tile_stride=self.args.tile_stride,
            disable_preprocessor=self.args.disable_preprocess_model,
            negative_prompt=self.args.negative_prompt, cfg_scale=self.args.cfg_scale
        )
        return output["samples"][0]


class BIDInferenceLoop(InferenceLoop):

    def build_pipeline(self) -> None:
        scunet = load_model("configs/model/scunet.yaml", "weights/scunet_psnr.ckpt").to(self.args.device).eval()
        controller = load_model("configs/model/cldm.yaml", "weights/controller.ckpt").to(self.args.device).eval()
        cond_fn = build_cond_fn(self.args)
        self.pipeline = BIDPipeline(scunet, controller, cond_fn)

    def restore_image(self, image: np.ndarray) -> np.ndarray:
        output = self.pipeline.run(
            image[None], steps=self.args.steps, strength=1.0, color_fix_type=self.args.color_fix_type,
            tiled=self.args.tiled, tile_size=self.args.tile_size, tile_stride=self.args.tile_stride,
            disable_preprocessor=self.args.disable_preprocess_model,
            negative_prompt=self.args.negative_prompt, cfg_scale=self.args.cfg_scale
        )
        return output["samples"][0]


class BFRInferenceLoop(InferenceLoop):
    
    def build_pipeline(self) -> None:
        swinir = load_model("configs/model/swinir.yaml", "weights/swinir.ckpt").to(self.args.device).eval()
        controller = load_model("configs/model/cldm.yaml", "weights/controller.ckpt").to(self.args.device).eval()
        cond_fn = build_cond_fn(self.args)
        self.pipeline = BFRPipeline(swinir, controller, cond_fn)

    def restore_image(self, image: np.ndarray) -> np.ndarray:
        output = self.pipeline.run(
            image[None], steps=self.args.steps, strength=1.0, color_fix_type=self.args.color_fix_type,
            tiled=self.args.tiled, tile_size=self.args.tile_size, tile_stride=self.args.tile_stride,
            disable_preprocessor=self.args.disable_preprocess_model,
            negative_prompt=self.args.negative_prompt, cfg_scale=self.args.cfg_scale
        )
        return output["samples"][0]


class UnAlignedBFRInferenceLoop(InferenceLoop):

    def build_pipeline(self) -> None:
        bsrnet = load_model("configs/model/bsrnet.yaml", "weights/bsrnet.ckpt").to(self.args.device).eval()
        swinir = load_model("configs/model/swinir.yaml", "weights/swinir.ckpt").to(self.args.device).eval()
        controller = load_model("configs/model/cldm.yaml", "weights/controller.ckpt").to(self.args.device).eval()
        cond_fn = build_cond_fn(self.args)
        self.bsr_pipeline = BSRPipeline(bsrnet, controller, cond_fn, upsample_scale=self.args.upsample_scale)
        self.bfr_pipeline = BFRPipeline(swinir, controller, cond_fn)
    
    def get_lq_generator(self) -> Generator:
        base_generator = super().get_lq_generator()
        self.face_helper = FaceRestoreHelper(
            device=self.args.device,
            upscale_factor=1,
            face_size=512,
            use_parse=True,
            det_model="retinaface_resnet50"
        )
        
        def _generator() -> np.ndarray:
            for image in base_generator():
                self.face_helper.clean_all()
                upsampled_bg = image_resize(image, self.args.upsample_scale)
                self.face_helper.read_image(upsampled_bg)
                # get face landmarks for each face
                self.face_helper.get_face_landmarks_5(resize=640, eye_dist_threshold=5)
                self.face_helper.align_warp_face()
                # restore cropped face images
                print(f"detect {len(self.face_helper.cropped_faces)} faces")
                for i, face_image in enumerate(self.face_helper.cropped_faces):
                    self.loop_context["face_idx"] = i
                    self.loop_context["is_face"] = True
                    self.loop_context["cropped_face"] = face_image
                    yield face_image
                # restore background
                self.loop_context["is_face"] = False
                yield image
        
        return _generator

    def setup(self) -> None:
        super().setup()
        self.cropped_face_dir = os.path.join(self.args.output, "cropped_faces")
        os.makedirs(self.cropped_face_dir, exist_ok=True)
        self.restored_face_dir = os.path.join(self.args.output, "restored_faces")
        os.makedirs(self.restored_face_dir, exist_ok=True)
        self.restored_bg_dir = os.path.join(self.args.output, "restored_backgrounds")
        os.makedirs(self.restored_bg_dir, exist_ok=True)

    def save_image(self, sample: np.ndarray) -> None:
        file_stem, repeat_idx = self.loop_context["file_stem"], self.loop_context["repeat_idx"]
        if self.loop_context["is_face"]:
            face_idx = self.loop_context["face_idx"]
            file_name = f"{file_stem}_{repeat_idx}_face_{face_idx}.png"
            Image.fromarray(sample).save(os.path.join(self.restored_face_dir, file_name))

            cropped_face = self.loop_context["cropped_face"]
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

    def restore_image(self, image: np.ndarray) -> np.ndarray:
        pipeline = self.bfr_pipeline if self.loop_context["is_face"] else self.bsr_pipeline
        output = pipeline.run(
            image[None], steps=self.args.steps, strength=1.0, color_fix_type=self.args.color_fix_type,
            tiled=self.args.tiled, tile_size=self.args.tile_size, tile_stride=self.args.tile_stride,
            disable_preprocessor=self.args.disable_preprocess_model,
            negative_prompt=self.args.negative_prompt, cfg_scale=self.args.cfg_scale
        )
        return output["samples"][0]


def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    # model parameters
    parser.add_argument("--task", type=str, required=True, choices=["sr", "dn", "fr", "fr_bg"])
    parser.add_argument("--upsample_scale", type=float, required=True)
    parser.add_argument("--version", type=str, default="v2")
    parser.add_argument("--disable_preprocess_model", action="store_true")
    parser.add_argument("--negative_prompt", type=str, default="low quality, blurry, low-resolution, noisy, unsharp, weird textures")
    parser.add_argument("--cfg_scale", type=float, default=4.0)
    # sampling parameters
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--tiled", action="store_true")
    parser.add_argument("--tile_size", type=int, default=512)
    parser.add_argument("--tile_stride", type=int, default=256)
    # input parameters
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--n_samples", type=int, default=1) # repeat_times
    # guidance parameters
    parser.add_argument("--use_guidance", action="store_true")
    parser.add_argument("--g_loss", type=str, default="mse", choices=["mse"])
    parser.add_argument("--g_scale", type=float, default=0.0)
    parser.add_argument("--g_t_start", type=int, default=1001)
    parser.add_argument("--g_t_stop", type=int, default=-1)
    parser.add_argument("--g_space", type=str, default="latent")
    parser.add_argument("--g_repeat", type=int, default=5)
    # output parameters
    parser.add_argument("--color_fix_type", type=str, default="wavelet", choices=["wavelet", "adain", "none"])
    parser.add_argument("--output", type=str, required=True)
    # common parameters
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--autocast", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    
    {
        "sr": BSRInferenceLoop,
        "dn": BIDInferenceLoop,
        "fr": BFRInferenceLoop,
        "fr_bg": UnAlignedBFRInferenceLoop
    }[args.task](args).run()


if __name__ == "__main__":
    main()
