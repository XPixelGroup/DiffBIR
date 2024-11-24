import os
from typing import Generator, List

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import pandas as pd

from .loop import InferenceLoop, MODELS
from ..utils.common import (
    instantiate_from_config,
    load_model_from_url,
    trace_vram_usage,
)
from ..utils.face import FaceRestoreHelper
from ..pipeline import (
    BSRNetPipeline,
    SwinIRPipeline,
)
from ..model import RRDBNet, SwinIR


class UnAlignedBFRInferenceLoop(InferenceLoop):

    def load_cleaner(self) -> None:
        if self.args.version == "v1":
            raise ValueError(
                "DiffBIR v1 doesn't support unaligned BFR, please use v2 or v2.1."
            )
        elif self.args.version == "v2":
            config = "configs/inference/bsrnet.yaml"
            weight = MODELS["bsrnet"]
        else:
            config = "configs/inference/swinir.yaml"
            weight = MODELS["swinir_realesrgan"]
        self.bg_cleaner: RRDBNet | SwinIR = instantiate_from_config(
            OmegaConf.load(config)
        )
        model_weight = load_model_from_url(weight)
        self.bg_cleaner.load_state_dict(model_weight, strict=True)
        self.bg_cleaner.eval().to(self.args.device)

        self.face_cleaner: SwinIR = instantiate_from_config(
            OmegaConf.load("configs/inference/swinir.yaml")
        )
        model_weight = load_model_from_url(MODELS["swinir_face"])
        self.face_cleaner.load_state_dict(model_weight, strict=True)
        self.face_cleaner.eval().to(self.args.device)

    def load_pipeline(self) -> None:
        if self.args.version == "v2":
            bg_pipeline = BSRNetPipeline(
                self.bg_cleaner,
                self.cldm,
                self.diffusion,
                self.cond_fn,
                self.args.device,
                self.args.upscale,
            )
            self.bg_requires_upscale = False
        else:
            # v2.1
            bg_pipeline = SwinIRPipeline(
                self.bg_cleaner,
                self.cldm,
                self.diffusion,
                self.cond_fn,
                self.args.device,
            )
            self.bg_requires_upscale = True
        self.pipeline_dict = {
            "background": bg_pipeline,
            "face": SwinIRPipeline(
                self.face_cleaner,
                self.cldm,
                self.diffusion,
                self.cond_fn,
                self.args.device,
            ),
        }

    def setup(self) -> None:
        super().setup()

        self.cropped_face_dir = os.path.join(self.save_dir, "cropped_faces")
        self.restored_face_dir = os.path.join(self.save_dir, "restored_faces")
        self.restored_bg_dir = os.path.join(self.save_dir, "restored_backgrounds")
        for dir_path in [
            self.cropped_face_dir,
            self.restored_face_dir,
            self.restored_bg_dir,
        ]:
            os.makedirs(dir_path, exist_ok=True)

        self.face_helper = FaceRestoreHelper(
            device=self.args.device,
            upscale_factor=1,
            face_size=512,
            use_parse=True,
            det_model="retinaface_resnet50",
        )
        self.face_samples = []

    def load_lq(self) -> Generator[Image.Image, None, None]:
        for lq in super().load_lq():
            self.face_helper.clean_all()
            self.face_samples.clear()

            upscaled_bg = np.array(
                lq.resize(
                    tuple(int(x * self.args.upscale) for x in lq.size), Image.BICUBIC
                )
            )
            self.face_helper.read_image(upscaled_bg)

            self.face_helper.get_face_landmarks_5(resize=640, eye_dist_threshold=5)
            self.face_helper.align_warp_face()
            print(f"detect {len(self.face_helper.cropped_faces)} faces")

            for i, lq_face in enumerate(self.face_helper.cropped_faces):
                self.loop_ctx["is_face"] = True
                self.loop_ctx["face_idx"] = i
                self.loop_ctx["cropped_face"] = lq_face
                yield Image.fromarray(lq_face)

            self.loop_ctx["is_face"] = False
            yield lq

    def after_load_lq(self, lq: Image.Image) -> np.ndarray:
        if self.loop_ctx["is_face"]:
            self.pipeline = self.pipeline_dict["face"]
        else:
            self.pipeline = self.pipeline_dict["background"]
            if self.bg_requires_upscale:
                lq = lq.resize(
                    tuple(int(x * self.args.upscale) for x in lq.size), Image.BICUBIC
                )
        return super().after_load_lq(lq)

    def save(self, samples: List[np.ndarray], pos_prompt: str, neg_prompt: str) -> None:
        file_stem = self.loop_ctx["file_stem"]
        # save prompt
        csv_path = os.path.join(self.save_dir, "prompt.csv")
        saved_file_stem = (
            f"{file_stem}_face_{self.loop_ctx['face_idx']}"
            if self.loop_ctx["is_face"]
            else file_stem
        )
        df = pd.DataFrame(
            {
                "file_name": [saved_file_stem],
                "pos_prompt": [pos_prompt],
                "neg_prompt": [neg_prompt],
            }
        )
        if os.path.exists(csv_path):
            df.to_csv(csv_path, index=None, mode="a", header=None)
        else:
            df.to_csv(csv_path, index=None)

        if self.loop_ctx["is_face"]:
            face_idx = self.loop_ctx["face_idx"]
            # save restored faces
            for i, sample in enumerate(samples):
                file_name = f"{file_stem}_face_{face_idx}_{i}.png"
                Image.fromarray(sample).save(
                    os.path.join(self.restored_face_dir, file_name)
                )
            # save cropped faces (lq)
            cropped_face = self.loop_ctx["cropped_face"]
            Image.fromarray(cropped_face).save(
                os.path.join(self.cropped_face_dir, file_name) # TODO: file_name is incorrect
            )
            # waiting for restored background image
            self.face_samples.append(samples)
        else:
            self.face_helper.get_inverse_affine()
            # transpose 2D list
            face_samples = list(map(list, zip(*self.face_samples)))
            for i, (restored_faces, restored_bg) in enumerate(
                zip(face_samples, samples)
            ):
                # add restored faces
                for face in restored_faces:
                    self.face_helper.add_restored_face(face)
                # paste each restored face to the input image
                restored_img = self.face_helper.paste_faces_to_input_image(
                    upsample_img=restored_bg
                )
                # save result
                file_name = f"{file_stem}_{i}.png"
                Image.fromarray(restored_bg).save(
                    os.path.join(self.restored_bg_dir, file_name)
                )
                Image.fromarray(restored_img).save(
                    os.path.join(self.save_dir, file_name)
                )
                # clear restored faces
                self.face_helper.restored_faces.clear()
