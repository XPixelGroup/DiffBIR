import os
import math
import torch
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
import pytorch_lightning as pl
from typing import List, Tuple
from argparse import ArgumentParser, Namespace

from facexlib.utils.face_restoration_helper import FaceRestoreHelper

from ldm.xformers_state import disable_xformers
from model.cldm import ControlLDM
from model.ddim_sampler import DDIMSampler
from model.spaced_sampler import SpacedSampler
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts
from utils.image import (
    wavelet_reconstruction, adaptive_instance_normalization, auto_resize, pad
)

from inference import process


def parse_args() -> Namespace:
    parser = ArgumentParser()
    # model
    parser.add_argument("--ckpt", required=True, type=str, help='Model checkpoint.')
    parser.add_argument("--config", required=True, type=str, help='Model config file.')
    parser.add_argument("--reload_swinir", action="store_true")
    parser.add_argument("--swinir_ckpt", type=str, default="")

    # input and preprocessing
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--sr_scale", type=float, default=1)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--disable_preprocess_model", action="store_true")

    # face related
    parser.add_argument('--has_aligned', action='store_true', help='Input are cropped and aligned faces. Default: False')
    parser.add_argument('--only_center_face', action='store_true', help='Only restore the center face. Default: False')
    parser.add_argument('--detection_model', type=str, default='retinaface_resnet50', 
            help='Face detector. Optional: retinaface_resnet50, retinaface_mobile0.25, YOLOv5l, YOLOv5n, dlib. \
                Default: retinaface_resnet50')
    # TODO: support diffbir background upsampler
    # parser.add_argument('--bg_upsampler', type=str, default='None', help='Background upsampler. Optional: diffbir, realesrgan')
    
    # postprocessing and saving
    parser.add_argument("--color_fix_type", type=str, default="wavelet", choices=["wavelet", "adain", "none"])
    parser.add_argument("--resize_back", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--show_lq", action="store_true")
    parser.add_argument("--skip_if_exist", action="store_true")
    
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    img_save_ext = 'png'
    pl.seed_everything(args.seed)
    
    if args.device == "cpu":
        disable_xformers()
    
    model: ControlLDM = instantiate_from_config(OmegaConf.load(args.config))
    load_state_dict(model, torch.load(args.ckpt, map_location="cpu"), strict=True)
    # reload preprocess model if specified
    if args.reload_swinir:
        if not hasattr(model, "preprocess_model"):
            raise ValueError(f"model don't have a preprocess model.")
        print(f"reload swinir model from {args.swinir_ckpt}")
        load_state_dict(model.preprocess_model, torch.load(args.swinir_ckpt, map_location="cpu"), strict=True)
    model.freeze()
    model.to(args.device)
    
    assert os.path.isdir(args.input)

    # ------------------ set up FaceRestoreHelper -------------------
    face_helper = FaceRestoreHelper(
        device=args.device, 
        upscale_factor=1, 
        face_size=args.image_size, 
        use_parse=True,
        det_model = args.detection_model
        )
    # TODO: to support backgrouns upsampler
    bg_upsampler = None
    
    print(f"sampling {args.steps} steps using {args.sampler} sampler")
    for file_path in list_image_files(args.input, follow_links=True):
        # read image
        lq = Image.open(file_path).convert("RGB")
        if args.sr_scale != 1:
            lq = lq.resize(
                tuple(math.ceil(x * args.sr_scale) for x in lq.size),
                Image.BICUBIC
            )
        lq_resized = auto_resize(lq, args.image_size)
        x = pad(np.array(lq_resized), scale=64)

        face_helper.clean_all()
        if args.has_aligned: 
            # the input faces are already cropped and aligned
            face_helper.cropped_faces = [x]
        else:
            face_helper.read_image(x)
            # get face landmarks for each face
            face_helper.get_face_landmarks_5(only_center_face=args.only_center_face, resize=640, eye_dist_threshold=5)
            face_helper.align_warp_face()

        save_path = os.path.join(args.output, os.path.relpath(file_path, args.input))
        parent_path, basename, _ = get_file_name_parts(save_path)
        os.makedirs(parent_path, exist_ok=True)
        os.makedirs(os.path.join(parent_path, 'cropped_faces'), exist_ok=True)
        os.makedirs(os.path.join(parent_path, 'restored_faces'), exist_ok=True)
        os.makedirs(os.path.join(parent_path, 'restored_imgs'), exist_ok=True)
        for i in range(args.repeat_times):
            restored_img_path = os.path.join(parent_path, 'restored_imgs', f'{basename}.{img_save_ext}')
            if os.path.exists(restored_img_path):
                if args.skip_if_exist:
                    print(f"Exists, skip face image {basename}...")
                    continue
                else:
                    raise RuntimeError(f"Image {basename} already exist")
            
            try:
                preds, stage1_preds = process(
                    model, face_helper.cropped_faces, steps=args.steps, sampler=args.sampler,
                    strength=1,
                    color_fix_type=args.color_fix_type,
                    disable_preprocess_model=args.disable_preprocess_model
                )
            except RuntimeError as e:
                # Avoid cuda_out_of_memory error.
                print(f"{file_path}, error: {e}")
                continue
            
            for restored_face in preds:
                # unused stage1 preds
                # face_helper.add_restored_face(np.array(stage1_restored_face))
                face_helper.add_restored_face(np.array(restored_face))

            # paste face back to the image
            if not args.has_aligned:
                # upsample the background
                if bg_upsampler is not None:
                    # TODO
                    bg_img = None
                else:
                    bg_img = None
                face_helper.get_inverse_affine(None)

                # paste each restored face to the input image
                restored_img = face_helper.paste_faces_to_input_image(
                    upsample_img=bg_img
                )

            # save faces
            for idx, (cropped_face, restored_face) in enumerate(zip(face_helper.cropped_faces, face_helper.restored_faces)):
                save_path = os.path.join(parent_path, f"{basename}_{i}.{img_save_ext}")
                # save cropped face
                if not args.has_aligned: 
                    save_crop_path = os.path.join(parent_path, 'cropped_faces', f'{basename}_{idx:02d}.{img_save_ext}')
                    Image.fromarray(cropped_face).save(save_crop_path)
                # save restored face
                if args.has_aligned:
                    save_face_name = f'{basename}.{img_save_ext}'
                else:
                    save_face_name = f'{basename}_{idx:02d}.{img_save_ext}'
                save_restore_path = os.path.join(parent_path, 'restored_faces', save_face_name)
                Image.fromarray(restored_face).save(save_restore_path)

            if not args.has_aligned:
                # remove padding
                restored_img = restored_img[:lq_resized.height, :lq_resized.width, :]
                # save restored image
                if args.resize_back and lq_resized.size != lq.size:
                    Image.fromarray(restored_img).resize(lq.size, Image.LANCZOS).convert("RGB").save(restored_img_path)
                else:
                    Image.fromarray(restored_img).convert("RGB").save(restored_img_path)
            print(f"Face image {basename} saved to {parent_path}")


if __name__ == "__main__":
    main()
