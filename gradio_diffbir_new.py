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

from model.cldm import ControlLDM
from utils.image import auto_resize, pad
from utils.common import instantiate_from_config, load_state_dict
from model.pipeline import BSRPipeline
from inference import load_model
from model.hacked_cldm import OptimizationFlag, hack_everything


hack_everything()

OptimizationFlag.enable_xformers()

parser = ArgumentParser()
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
args = parser.parse_args()

# load model
bsrnet = load_model("configs/model/bsrnet.yaml", "weights/bsrnet.ckpt").to(args.device).eval()
controller = load_model("configs/model/cldm.yaml", "weights/controller.ckpt").to(args.device).eval()
# pipeline = BSRPipeline(bsrnet, controller, cond_fn, upsample_scale=upsample_scale)


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
    # tiled: bool,
    # tile_size: int,
    # tile_stride: int,
    fp16,
    deepcache,
    cache_block_index,
    cache_interval,
    cache_uniform,
    progress = gr.Progress(track_tqdm=True)
) -> List[np.ndarray]:
    if fp16 and not OptimizationFlag.is_fp16_enabled():
        OptimizationFlag.enable_fp16()
        print("switch to fp16 mode")
        controller.to_fp16()
    elif not fp16 and OptimizationFlag.is_fp16_enabled():
        OptimizationFlag.disable_fp16()
        print("switch to fp32 mode")
        controller.to_fp32()

    if deepcache:
        OptimizationFlag.enable_deepcache(cache_block_index, cache_interval, cache_uniform, 1.4, 15)
    else:
        OptimizationFlag.disable_deepcache()

    # print(
    #     f"control image shape={control_img.size}\n"
    #     f"num_samples={num_samples}, sr_scale={sr_scale}\n"
    #     f"disable_preprocess_model={disable_preprocess_model}, strength={strength}\n"
    #     f"positive_prompt='{positive_prompt}', negative_prompt='{negative_prompt}'\n"
    #     f"cdf scale={cfg_scale}, steps={steps}, use_color_fix={use_color_fix}\n"
    #     f"seed={seed}\n"
    #     # f"tiled={tiled}, tile_size={tile_size}, tile_stride={tile_stride}"
    # )
    pl.seed_everything(seed)
    
    pipeline = BSRPipeline(bsrnet, controller, None, upsample_scale=sr_scale)
    
    samples = []
    for _ in tqdm(range(num_samples)):
        sample = pipeline.run(
            control_img[None], steps=steps, strength=strength, color_fix_type="wavelet" if use_color_fix else "none",
            tiled=False, tile_size=512, tile_stride=256,
            disable_preprocessor=disable_preprocess_model,
            positive_prompt=positive_prompt,
            negative_prompt=negative_prompt, cfg_scale=cfg_scale
        )["samples"][0]
        samples.append(sample)
    
    return samples


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
                # tiled = gr.Checkbox(label="Tiled", value=False)
                # tile_size = gr.Slider(label="Tile Size", minimum=512, maximum=1024, value=512, step=256)
                # tile_stride = gr.Slider(label="Tile Stride", minimum=256, maximum=512, value=256, step=128)
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
                fp16 = gr.Checkbox(label="Half-precision mode", value=False)
                deepcache = gr.Checkbox(label="Enable DeepCache", value=False)
                cache_block_index = gr.Slider(label="Cache Block Index", minimum=0, maximum=3, value=1, step=1)
                cache_interval = gr.Slider(label="Cache Interval", minimum=1, maximum=10, value=5, step=1)
                cache_uniform = gr.Checkbox(label="Cache Uniform", value=False)
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
        # tiled,
        # tile_size,
        # tile_stride
        fp16,
        deepcache,
        cache_block_index,
        cache_interval,
        cache_uniform
    ]
    run_button.click(fn=process, inputs=inputs, outputs=[result_gallery])

block.launch(share=True)
