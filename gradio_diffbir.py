from typing import List
import math
from argparse import ArgumentParser

import numpy as np
import torch
import einops
import pytorch_lightning as pl
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf

from ldm.xformers_state import disable_xformers
from model.spaced_sampler import SpacedSampler
from model.cldm import ControlLDM
from utils.image import (
    wavelet_reconstruction, auto_resize, pad
)
from utils.common import instantiate_from_config, load_state_dict


parser = ArgumentParser()
parser.add_argument("--config", required=True, type=str)
parser.add_argument("--ckpt", type=str, required=True)
parser.add_argument("--reload_swinir", action="store_true")
parser.add_argument("--swinir_ckpt", type=str, default="")
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
args = parser.parse_args()

# load model
if args.device == "cpu":
    disable_xformers()
model: ControlLDM = instantiate_from_config(OmegaConf.load(args.config))
load_state_dict(model, torch.load(args.ckpt, map_location="cpu"), strict=True)
# reload preprocess model if specified
if args.reload_swinir:
    print(f"reload swinir model from {args.swinir_ckpt}")
    load_state_dict(model.preprocess_model, torch.load(args.swinir_ckpt, map_location="cpu"), strict=True)
model.freeze()
model.to(args.device)
# load sampler
sampler = SpacedSampler(model, var_type="fixed_small")


@torch.no_grad()
def process(
    control_img: Image.Image,
    num_samples: int,
    sr_scale: int,
    image_size: int,
    disable_preprocess_model: bool,
    strength: float,
    positive_prompt: str,
    negative_prompt: str,
    cond_scale: float,
    steps: int,
    use_color_fix: bool,
    keep_original_size: bool,
    seed: int
) -> List[np.ndarray]:
    print(
        f"control image shape={control_img.size}\n"
        f"num_samples={num_samples}, sr_scale={sr_scale}, image_size={image_size}\n"
        f"disable_preprocess_model={disable_preprocess_model}, strength={strength}\n"
        f"positive_prompt='{positive_prompt}', negative_prompt='{negative_prompt}'\n"
        f"prompt scale={cond_scale}, steps={steps}, use_color_fix={use_color_fix}\n"
        f"seed={seed}"
    )
    pl.seed_everything(seed)

    # prepare condition
    if sr_scale != 1:
        control_img = control_img.resize(
            tuple(math.ceil(x * sr_scale) for x in control_img.size),
            Image.BICUBIC
        )
    input_size = control_img.size
    control_img = auto_resize(control_img, image_size)
    h, w = control_img.height, control_img.width
    control_img = pad(np.array(control_img), scale=64) # HWC, RGB, [0, 255]
    control_imgs = [control_img] * num_samples
    control = torch.tensor(np.stack(control_imgs) / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    if not disable_preprocess_model:
        control = model.preprocess_model(control)
    height, width = control.size(-2), control.size(-1)
    cond = {
        "c_latent": [model.apply_condition_encoder(control)],
        "c_crossattn": [model.get_learned_conditioning([positive_prompt] * num_samples)]
    }
    uncond = {
        "c_latent": [model.apply_condition_encoder(control)],
        "c_crossattn": [model.get_learned_conditioning([negative_prompt] * num_samples)]
    }
    model.control_scales = [strength] * 13
    
    shape = (num_samples, 4, height // 8, width // 8)
    print(f"latent shape = {shape}")
    x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
    samples = sampler.sample(
        steps, shape, cond,
        unconditional_guidance_scale=cond_scale,
        unconditional_conditioning=uncond,
        cond_fn=None, x_T=x_T
    )
    x_samples = model.decode_first_stage(samples)
    x_samples = ((x_samples + 1) / 2).clamp(0, 1)
    
    # apply color correction
    if use_color_fix:
        x_samples = wavelet_reconstruction(x_samples, control)
    
    x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
    preds = []
    for img in x_samples:
        if keep_original_size:
            # remove padding and resize to input size
            img = Image.fromarray(img[:h, :w, :]).resize(input_size, Image.LANCZOS)
            preds.append(np.array(img))
        else:
            # remove padding
            preds.append(img[:h, :w, :])
    return preds


block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown("## DiffBIR")
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", type="pil")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Options", open=True):
                num_samples = gr.Slider(label="Images", minimum=1, maximum=12, value=1, step=1)
                sr_scale = gr.Number(label="SR Scale", value=1)
                image_size = gr.Slider(label="Image size", minimum=256, maximum=768, value=512, step=64)
                positive_prompt = gr.Textbox(label="Positive Prompt", value="")
                # It's worth noting that if your positive prompt is short while the negative prompt 
                # is long, the positive prompt will lose its effectiveness.
                # Example (control strength = 0):
                # positive prompt: cat
                # negative prompt: longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality
                # I take some experiments and find that sd_v2.1 will suffer from this problem while sd_v1.5 won't.
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
                )
                cond_scale = gr.Slider(label="Prompt Guidance Scale", minimum=0.1, maximum=30.0, value=1.0, step=0.1)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                disable_preprocess_model = gr.Checkbox(label="Disable Preprocess Model", value=False)
                use_color_fix = gr.Checkbox(label="Use Color Correction", value=True)
                keep_original_size = gr.Checkbox(label="Keep Original Size", value=True)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=231)
        with gr.Column():
            result_gallery = gr.Gallery(label="Output", show_label=False, elem_id="gallery").style(grid=2, height="auto")
    inputs = [
        input_image,
        num_samples,
        sr_scale,
        image_size,
        disable_preprocess_model,
        strength,
        positive_prompt,
        negative_prompt,
        cond_scale,
        steps,
        use_color_fix,
        keep_original_size,
        seed
    ]
    run_button.click(fn=process, inputs=inputs, outputs=[result_gallery])

block.launch(server_name='0.0.0.0')
