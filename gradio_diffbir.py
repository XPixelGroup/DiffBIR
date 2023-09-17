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
from tqdm import tqdm

from ldm.xformers_state import disable_xformers
from model.spaced_sampler import SpacedSampler
from model.cldm import ControlLDM
from utils.image import auto_resize, pad
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
    disable_preprocess_model: bool,
    strength: float,
    positive_prompt: str,
    negative_prompt: str,
    cfg_scale: float,
    steps: int,
    use_color_fix: bool,
    seed: int,
    tiled: bool,
    tile_size: int,
    tile_stride: int,
    progress = gr.Progress(track_tqdm=True)
) -> List[np.ndarray]:
    print(
        f"control image shape={control_img.size}\n"
        f"num_samples={num_samples}, sr_scale={sr_scale}\n"
        f"disable_preprocess_model={disable_preprocess_model}, strength={strength}\n"
        f"positive_prompt='{positive_prompt}', negative_prompt='{negative_prompt}'\n"
        f"cdf scale={cfg_scale}, steps={steps}, use_color_fix={use_color_fix}\n"
        f"seed={seed}\n"
        f"tiled={tiled}, tile_size={tile_size}, tile_stride={tile_stride}"
    )
    pl.seed_everything(seed)
    
    # resize lq
    if sr_scale != 1:
        control_img = control_img.resize(
            tuple(math.ceil(x * sr_scale) for x in control_img.size),
            Image.BICUBIC
        )
    
    # we regard the resized lq as the "original" lq and save its size for 
    # resizing back after restoration
    input_size = control_img.size
    
    if not tiled:
        # if tiled is not specified, that is, directly use the lq as input, we just 
        # resize lq to a size >= 512 since DiffBIR is trained on a resolution of 512
        control_img = auto_resize(control_img, 512)
    else:
        # otherwise we size lq to a size >= tile_size to ensure that the image can be 
        # divided into as least one patch
        control_img = auto_resize(control_img, tile_size)
    # save size for removing padding
    h, w = control_img.height, control_img.width
    
    # pad image to be multiples of 64
    control_img = pad(np.array(control_img), scale=64) # HWC, RGB, [0, 255]
    
    # convert to tensor (NCHW, [0,1])
    control = torch.tensor(control_img[None] / 255.0, dtype=torch.float32, device=model.device).clamp_(0, 1)
    control = einops.rearrange(control, "n h w c -> n c h w").contiguous()
    if not disable_preprocess_model:
        control = model.preprocess_model(control)
    height, width = control.size(-2), control.size(-1)
    model.control_scales = [strength] * 13
    
    preds = []
    for _ in tqdm(range(num_samples)):
        shape = (1, 4, height // 8, width // 8)
        x_T = torch.randn(shape, device=model.device, dtype=torch.float32)
        if not tiled:
            samples = sampler.sample(
                steps=steps, shape=shape, cond_img=control,
                positive_prompt=positive_prompt, negative_prompt=negative_prompt, x_T=x_T,
                cfg_scale=cfg_scale, cond_fn=None,
                color_fix_type="wavelet" if use_color_fix else "none"
            )
        else:
            samples = sampler.sample_with_mixdiff(
                tile_size=int(tile_size), tile_stride=int(tile_stride),
                steps=steps, shape=shape, cond_img=control,
                positive_prompt=positive_prompt, negative_prompt=negative_prompt, x_T=x_T,
                cfg_scale=cfg_scale, cond_fn=None,
                color_fix_type="wavelet" if use_color_fix else "none"
            )
        x_samples = samples.clamp(0, 1)
        x_samples = (einops.rearrange(x_samples, "b c h w -> b h w c") * 255).cpu().numpy().clip(0, 255).astype(np.uint8)
        # remove padding and resize to input size
        img = Image.fromarray(x_samples[0, :h, :w, :]).resize(input_size, Image.LANCZOS)
        preds.append(np.array(img))
    
    return preds

MARKDOWN = \
"""
## DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior

[GitHub](https://github.com/XPixelGroup/DiffBIR) | [Paper](https://arxiv.org/abs/2308.15070) | [Project Page](https://0x3f3f3f3fun.github.io/projects/diffbir/)

If DiffBIR is helpful for you, please help star the GitHub Repo. Thanks!
"""

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(source="upload", type="pil")
            run_button = gr.Button(label="Run")
            with gr.Accordion("Options", open=True):
                tiled = gr.Checkbox(label="Tiled", value=False)
                tile_size = gr.Slider(label="Tile Size", minimum=512, maximum=1024, value=512, step=256)
                tile_stride = gr.Slider(label="Tile Stride", minimum=256, maximum=512, value=256, step=128)
                num_samples = gr.Slider(label="Number Of Samples", minimum=1, maximum=12, value=1, step=1)
                sr_scale = gr.Number(label="SR Scale", value=1)
                positive_prompt = gr.Textbox(label="Positive Prompt", value="")
                negative_prompt = gr.Textbox(
                    label="Negative Prompt",
                    value="longbody, lowres, bad anatomy, bad hands, missing fingers, extra digit, fewer digits, cropped, worst quality, low quality"
                )
                cfg_scale = gr.Slider(label="Classifier Free Guidance Scale (Set a value larger than 1 to enable it!)", minimum=0.1, maximum=30.0, value=1.0, step=0.1)
                strength = gr.Slider(label="Control Strength", minimum=0.0, maximum=2.0, value=1.0, step=0.01)
                steps = gr.Slider(label="Steps", minimum=1, maximum=100, value=50, step=1)
                disable_preprocess_model = gr.Checkbox(label="Disable Preprocess Model", value=False)
                use_color_fix = gr.Checkbox(label="Use Color Correction", value=True)
                seed = gr.Slider(label="Seed", minimum=-1, maximum=2147483647, step=1, value=231)
        with gr.Column():
            result_gallery = gr.Gallery(label="Output", show_label=False, elem_id="gallery").style(grid=2, height="auto")
    inputs = [
        input_image,
        num_samples,
        sr_scale,
        disable_preprocess_model,
        strength,
        positive_prompt,
        negative_prompt,
        cfg_scale,
        steps,
        use_color_fix,
        seed,
        tiled,
        tile_size,
        tile_stride
    ]
    run_button.click(fn=process, inputs=inputs, outputs=[result_gallery])

block.launch()
