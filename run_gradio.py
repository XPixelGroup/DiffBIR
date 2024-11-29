from typing import List
import math
from argparse import ArgumentParser
import random

import numpy as np
import torch
import gradio as gr
from PIL import Image
from omegaconf import OmegaConf
from tqdm import tqdm
from accelerate.utils import set_seed

from diffbir.model.cldm import ControlLDM
from diffbir.model.swinir import SwinIR
from diffbir.inference.pretrained_models import MODELS
from diffbir.utils.common import instantiate_from_config, load_model_from_url
from diffbir.model.gaussian_diffusion import Diffusion
from diffbir.pipeline import SwinIRPipeline
from diffbir.utils.caption import (
    EmptyCaptioner,
    LLaVACaptioner,
    RAMCaptioner,
    LLAVA_AVAILABLE,
    RAM_AVAILABLE,
)

torch.set_grad_enabled(False)

# This gradio script only support DiffBIR v2.1
parser = ArgumentParser()
parser.add_argument("--captioner", type=str, choices=["none", "ram", "llava"], required=True)
parser.add_argument("--llava_bit", type=str, choices=["4", "8", "16"], default="4")
args = parser.parse_args()

# Set max height and width to constraint inference time for online demo
max_height = 2048
max_width = 2048

tasks = ["sr", "face"]
device = "cuda"
precision = "fp16"
llava_bit = args.llava_bit
# Set captioner to llava or ram to enable auto-caption
captioner = args.captioner

if captioner == "llava":
    assert LLAVA_AVAILABLE
elif captioner == "ram":
    assert RAM_AVAILABLE

# 1. load stage-1 models
swinir: SwinIR = instantiate_from_config(
    OmegaConf.load("configs/inference/swinir.yaml")
)
swinir.load_state_dict(load_model_from_url(MODELS["swinir_realesrgan"]))
swinir.eval().to(device)

face_swinir: SwinIR = instantiate_from_config(
    OmegaConf.load("configs/inference/swinir.yaml")
)
face_swinir.load_state_dict(load_model_from_url(MODELS["swinir_face"]))
face_swinir.eval().to(device)

# 2. load stage-2 model
cldm: ControlLDM = instantiate_from_config(
    OmegaConf.load("configs/inference/cldm.yaml")
)
# 2.1 load pre-trained SD
sd_weight = load_model_from_url(MODELS["sd_v2.1_zsnr"])
unused, missing = cldm.load_pretrained_sd(sd_weight)
print(
    f"load pretrained stable diffusion, "
    f"unused weights: {unused}, missing weights: {missing}"
)
# 2.2 load ControlNet
control_weight = load_model_from_url(MODELS["v2.1"])
cldm.load_controlnet_from_ckpt(control_weight)
print("load controlnet weight")
cldm.eval().to(device)
cast_type = {
    "fp32": torch.float32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}[precision]
cldm.cast_dtype(cast_type)

# 3. load noise schedule
diffusion: Diffusion = instantiate_from_config(
    OmegaConf.load("configs/inference/diffusion_v2.1.yaml")
)
diffusion.to(device)

# 4. load captioner
if captioner == "none":
    captioner = EmptyCaptioner(device)
elif captioner == "llava":
    captioner = LLaVACaptioner(device, llava_bit)
else:
    captioner = RAMCaptioner(device)

error_image = np.array(Image.open("assets/gradio_error_img.png"))


@torch.no_grad()
def process(
    input_image,
    task,
    upscale,
    cleaner_tiled,
    cleaner_tile_size,
    vae_encoder_tiled,
    vae_encoder_tile_size,
    vae_decoder_tiled,
    vae_decoder_tile_size,
    cldm_tiled,
    cldm_tile_size,
    positive_prompt,
    negative_prompt,
    cfg_scale,
    rescale_cfg,
    strength,
    noise_aug,
    steps,
    sampler_type,
    s_churn,
    s_tmin,
    s_tmax,
    s_noise,
    order,
    seed,
    progress=gr.Progress(track_tqdm=True),
) -> List[np.ndarray]:
    if seed == -1:
        seed = random.randint(0, 2147483647)
    set_seed(seed)
    lq = input_image
    # Prepare prompt
    caption = captioner(lq)
    pos_prompt = ", ".join([text for text in [caption, positive_prompt] if text])
    neg_prompt = negative_prompt
    # Upscale and convert to numpy array
    out_w, out_h = tuple(int(x * upscale) for x in lq.size)
    if out_w > max_width or out_h > max_height:
        return [error_image], (
            "Failed :( The requested resolution exceeds the maximum limit. "
            f"Your requested resolution is ({out_h}, {out_w}). "
            f"The maximum allowed resolution is ({max_height}, {max_width})."
        )
    lq = lq.resize((out_w, out_h), Image.BICUBIC)
    lq = np.array(lq)
    # Select cleaner
    if task == "sr":
        cleaner = swinir
    else:
        cleaner = face_swinir
    # Create pipeline
    pipeline = SwinIRPipeline(cleaner, cldm, diffusion, None, device)
    # Run pipeline to restore this image
    try:
        sample = pipeline.run(
            lq[None],
            steps,
            strength,
            cleaner_tiled,
            cleaner_tile_size,
            cleaner_tile_size // 2,
            vae_encoder_tiled,
            vae_encoder_tile_size,
            vae_decoder_tiled,
            vae_decoder_tile_size,
            cldm_tiled,
            cldm_tile_size,
            cldm_tile_size // 2,
            pos_prompt,
            neg_prompt,
            cfg_scale,
            "noise",
            sampler_type,
            noise_aug,
            rescale_cfg,
            s_churn,
            s_tmin,
            s_tmax,
            s_noise,
            1,
            order,
        )[0]
        return [sample], "Success :)"
    except Exception as e:
        return [error_image], f"Failed :( {e}"


# TODO: add help information for each option
MARKDOWN = """
## DiffBIR: Towards Blind Image Restoration with Generative Diffusion Prior

[GitHub](https://github.com/XPixelGroup/DiffBIR) | [Paper](https://arxiv.org/abs/2308.15070) | [Project Page](https://0x3f3f3f3fun.github.io/projects/diffbir/)

If DiffBIR is helpful for you, please help star the GitHub Repo. Thanks!
"""

DEFAULT_POS_PROMPT = (
    "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
    "hyper detailed photo - realistic maximum detail, 32k, Color Grading, ultra HD, extreme meticulous detailing, "
    "skin pore detailing, hyper sharpness, perfect without deformations."
)

DEFAULT_NEG_PROMPT = (
    "painting, oil painting, illustration, drawing, art, sketch, oil painting, cartoon, "
    "CG Style, 3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, watermark, "
    "signature, jpeg artifacts, deformed, lowres, over-smooth."
)

block = gr.Blocks().queue()
with block:
    with gr.Row():
        gr.Markdown(MARKDOWN)
    with gr.Row():
        with gr.Column():
            input_image = gr.Image(sources="upload", type="pil")
            run_button = gr.Button(value="Run")
            with gr.Accordion("Basic Options", open=True):
                with gr.Row():
                    task = gr.Dropdown(
                        label="Task",
                        choices=tasks,
                        value="sr",
                    )
                    upscale = gr.Slider(
                        label="Upsample factor",
                        minimum=1,
                        maximum=8,
                        value=4,
                        step=1,
                    )
                with gr.Row():
                    with gr.Column():
                        cleaner_tiled = gr.Checkbox(
                            label="Tiled cleaner",
                            value=False,
                        )
                        cleaner_tile_size = gr.Slider(
                            label="Cleaner tile size",
                            minimum=256,
                            maximum=1024,
                            value=256,
                            step=64,
                        )
                    with gr.Column():
                        vae_encoder_tiled = gr.Checkbox(
                            label="Tiled VAE encoder",
                            value=False,
                        )
                        vae_encoder_tile_size = gr.Slider(
                            label="VAE encoder tile size",
                            minimum=256,
                            maximum=1024,
                            value=256,
                            step=64,
                        )
                with gr.Row():
                    with gr.Column():
                        vae_decoder_tiled = gr.Checkbox(
                            label="Tiled VAE decoder",
                            value=False,
                        )
                        vae_decoder_tile_size = gr.Slider(
                            label="VAE decoder tile size",
                            minimum=256,
                            maximum=1024,
                            value=256,
                            step=64,
                        )
                    with gr.Column():
                        cldm_tiled = gr.Checkbox(
                            label="Tiled diffusion",
                            value=True,
                        )
                        cldm_tile_size = gr.Slider(
                            label="Diffusion tile size",
                            minimum=512,
                            maximum=1024,
                            value=512,
                            step=64,
                        )
                seed = gr.Slider(
                    label="Seed", minimum=-1, maximum=2147483647, step=1, value=231
                )
            with gr.Accordion("Condition Options", open=True):
                pos_prompt = gr.Textbox(
                    label="Positive prompt",
                    value=DEFAULT_POS_PROMPT,
                )
                neg_prompt = gr.Textbox(
                    label="Negative prompt",
                    value=DEFAULT_NEG_PROMPT,
                )
                cfg_scale = gr.Slider(
                    label="Classifier-free guidance (cfg) scale",
                    minimum=1,
                    maximum=10,
                    value=8,
                    step=1,
                )
                rescale_cfg = gr.Checkbox(value=False, label="Gradually increase cfg scale")
                with gr.Row():
                    strength = gr.Slider(
                        label="Control strength",
                        minimum=0.0,
                        maximum=1.5,
                        value=1.0,
                        step=0.1,
                    )
                    noise_aug = gr.Slider(
                        label="Noise level of condition",
                        minimum=0,
                        maximum=199,
                        value=0,
                        step=10,
                    )
            with gr.Accordion("Sampler Options", open=True):
                steps = gr.Slider(
                    label="Steps", minimum=5, maximum=50, value=10, step=5
                )
                sampler_type = gr.Dropdown(
                    label="Select a sampler",
                    choices=[
                        "dpm++_m2",
                        "spaced",
                        "ddim",
                        "edm_euler",
                        "edm_euler_a",
                        "edm_heun",
                        "edm_dpm_2",
                        "edm_dpm_2_a",
                        "edm_lms",
                        "edm_dpm++_2s_a",
                        "edm_dpm++_sde",
                        "edm_dpm++_2m",
                        "edm_dpm++_2m_sde",
                        "edm_dpm++_3m_sde",
                    ],
                    value="edm_dpm++_3m_sde",
                )
                s_churn = gr.Slider(
                    label="s_churn",
                    minimum=0,
                    maximum=40,
                    value=0,
                    step=1,
                )
                s_tmin = gr.Slider(
                    label="s_tmin",
                    minimum=0,
                    maximum=300,
                    value=0,
                    step=10,
                )
                s_tmax = gr.Slider(
                    label="s_tmax",
                    minimum=0,
                    maximum=300,
                    value=300,
                    step=10,
                )
                s_noise = gr.Slider(
                    label="s_noise",
                    minimum=1,
                    maximum=1.1,
                    value=1,
                    step=0.001,
                )
                order = gr.Slider(
                    label="order",
                    minimum=1,
                    maximum=8,
                    value=1,
                    step=1,
                )
        with gr.Column():
            result_gallery = gr.Gallery(
                label="Output", show_label=False, columns=2, format="png"
            )
            status = gr.Textbox(label="Status")
    run_button.click(
        fn=process,
        inputs=[
            input_image,
            task,
            upscale,
            cleaner_tiled,
            cleaner_tile_size,
            vae_encoder_tiled,
            vae_encoder_tile_size,
            vae_decoder_tiled,
            vae_decoder_tile_size,
            cldm_tiled,
            cldm_tile_size,
            pos_prompt,
            neg_prompt,
            cfg_scale,
            rescale_cfg,
            strength,
            noise_aug,
            steps,
            sampler_type,
            s_churn,
            s_tmin,
            s_tmax,
            s_noise,
            order,
            seed,
        ],
        outputs=[result_gallery, status],
    )

block.launch()
