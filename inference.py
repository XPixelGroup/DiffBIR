from typing import List, Tuple
import os
import math
from argparse import ArgumentParser, Namespace

import numpy as np
import torch
import einops
import pytorch_lightning as pl
from PIL import Image
from omegaconf import OmegaConf

from ldm.xformers_state import disable_xformers
from model.spaced_sampler import SpacedSampler
from model.ddim_sampler import DDIMSampler
from model.cldm import ControlLDM
from utils.image import (
    wavelet_reconstruction, adaptive_instance_normalization, auto_resize, pad
)
from utils.common import instantiate_from_config, load_state_dict
from utils.file import list_image_files, get_file_name_parts


@torch.no_grad()
def process(
    model: ControlLDM,
    control_imgs: List[np.ndarray],
    sampler: str,
    steps: int,
    strength: float,
    color_fix_type: str,
    disable_preprocess_model: bool
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply DiffBIR model on a list of low-quality images.
    
    Args:
        model (ControlLDM): Model.
        control_imgs (List[np.ndarray]): A list of low-quality images (HWC, RGB, range in [0, 255])
        sampler (str): Sampler name.
        steps (int): Sampling steps.
        strength (float): Control strength. Set to 1.0 during training.
        color_fix_type (str): Type of color correction for samples.
        disable_preprocess_model (bool): If specified, preprocess model (SwinIR) will not be used.
    
    Returns:
        preds (List[np.ndarray]): Restoration results (HWC, RGB, range in [0, 255]).
        stage1_preds (List[np.ndarray]): Outputs of preprocess model (HWC, RGB, range in [0, 255]). 
            If `disable_preprocess_model` is specified, then preprocess model's outputs is the same 
            as low-quality inputs.
    """
    n_samples = len(control_imgs)
    if sampler == "ddpm":
        sampler = SpacedSampler(model, var_type="fixed_small")
    else:
        sampler = DDIMSampler(model)
    control = torch.tensor(np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    # TODO: model.preprocess_model = lambda x: x
    if not disable_preprocess_model and hasattr(model, "preprocess_model"):
        control = model.preprocess_model(control)
    elif disable_preprocess_model and not hasattr(model, "preprocess_model"):
        raise ValueError(f"model doesn't have a preprocess model.")
    
    height, width = control.size(-2), control.size(-1)
    cond = {
        "c_latent": [model.apply_condition_encoder(control)],
        "c_crossattn": [model.get_learned_conditioning([""] * n_samples)]
    }
    model.control_scales = [strength] * 13
    
    shape = (n_samples, 4, height // 8, width // 8)
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
    if isinstance(sampler, SpacedSampler):
        samples = sampler.sample(
            steps, shape, cond,
            unconditional_guidance_scale=1.0,
            unconditional_conditioning=None,
            cond_fn=None, x_T=x_T
        )
    else:
        sampler: DDIMSampler
        samples, _ = sampler.sample(
            S=steps, batch_size=shape[0], shape=shape[1:],
            conditioning=cond, unconditional_conditioning=None,
            x_T=x_T, eta=0
        )
    x_samples = model.decode_first_stage(samples)
    x_samples = ((x_samples + 1) / 2).clamp(0, 1)
    
    # apply color correction (borrowed from StableSR)
    if color_fix_type == "adain":
        x_samples = adaptive_instance_normalization(x_samples, control)
    elif color_fix_type == "wavelet":
        x_samples = wavelet_reconstruction(x_samples, control)
    else:
        assert color_fix_type == "none", f"unexpected color fix type: {color_fix_type}"
    
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    control = (einops.rearrange(control, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    
    preds = [x_samples[i] for i in range(n_samples)]
    stage1_preds = [control[i] for i in range(n_samples)]
    
    return preds, stage1_preds


def parse_args() -> Namespace:
    parser = ArgumentParser()
    
    parser.add_argument("--ckpt", required=True, type=str)
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--reload_swinir", action="store_true")
    parser.add_argument("--swinir_ckpt", type=str, default="")
    
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--sampler", type=str, default="ddpm", choices=["ddpm", "ddim"])
    parser.add_argument("--steps", required=True, type=int)
    parser.add_argument("--sr_scale", type=float, default=1)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--repeat_times", type=int, default=1)
    parser.add_argument("--disable_preprocess_model", action="store_true")
    
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
    
    print(f"sampling {args.steps} steps using {args.sampler} sampler")
    # with torch.autocast(device, dtype=torch.bfloat16):
    for file_path in list_image_files(args.input, follow_links=True):
        lq = Image.open(file_path).convert("RGB")
        if args.sr_scale != 1:
            lq = lq.resize(
                tuple(math.ceil(x * args.sr_scale) for x in lq.size),
                Image.BICUBIC
            )
        lq_resized = auto_resize(lq, args.image_size)
        x = pad(np.array(lq_resized), scale=64)
        
        for i in range(args.repeat_times):
            save_path = os.path.join(args.output, os.path.relpath(file_path, args.input))
            parent_path, stem, _ = get_file_name_parts(save_path)
            save_path = os.path.join(parent_path, f"{stem}_{i}.png")
            if os.path.exists(save_path):
                if args.skip_if_exist:
                    print(f"skip {save_path}")
                    continue
                else:
                    raise RuntimeError(f"{save_path} already exist")
            os.makedirs(parent_path, exist_ok=True)
            
            # try:
            preds, stage1_preds = process(
                model, [x], steps=args.steps, sampler=args.sampler,
                strength=1,
                color_fix_type=args.color_fix_type,
                disable_preprocess_model=args.disable_preprocess_model
            )
            # except RuntimeError as e:
            #     # Avoid cuda_out_of_memory error.
            #     print(f"{file_path}, error: {e}")
            #     continue
            
            pred, stage1_pred = preds[0], stage1_preds[0]
            
            # remove padding
            pred = pred[:lq_resized.height, :lq_resized.width, :]
            stage1_pred = stage1_pred[:lq_resized.height, :lq_resized.width, :]
            
            if args.show_lq:
                if args.resize_back:
                    if lq_resized.size != lq.size:
                        pred = np.array(Image.fromarray(pred).resize(lq.size, Image.LANCZOS))
                        stage1_pred = np.array(Image.fromarray(stage1_pred).resize(lq.size, Image.LANCZOS))
                    lq = np.array(lq)
                else:
                    lq = np.array(lq_resized)
                images = [lq, pred] if args.disable_preprocess_model else [lq, stage1_pred, pred]
                Image.fromarray(np.concatenate(images, axis=1)).save(save_path)
            else:
                if args.resize_back and lq_resized.size != lq.size:
                    Image.fromarray(pred).resize(lq.size, Image.LANCZOS).save(save_path)
                else:
                    Image.fromarray(pred).save(save_path)
            print(f"save to {save_path}")

if __name__ == "__main__":
    main()
