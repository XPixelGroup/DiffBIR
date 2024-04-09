import os
from argparse import ArgumentParser

from omegaconf import OmegaConf
import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from PIL import Image, ImageDraw, ImageFont
import numpy as np

from model import ControlLDM, SwinIR, Diffusion
from utils.common import instantiate_from_config
from utils.sampler import SpacedSampler


def log_txt_as_img(wh, xc):
    # wh a tuple of (width, height)
    # xc a list of captions to plot
    b = len(xc)
    txts = list()
    for bi in range(b):
        txt = Image.new("RGB", wh, color="white")
        draw = ImageDraw.Draw(txt)
        # font = ImageFont.truetype('font/DejaVuSans.ttf', size=size)
        font = ImageFont.load_default()
        nc = int(40 * (wh[0] / 256))
        lines = "\n".join(xc[bi][start:start + nc] for start in range(0, len(xc[bi]), nc))

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def main(args) -> None:
    # Setup accelerator:
    accelerator = Accelerator(split_batches=True)
    set_seed(231)
    device = accelerator.device
    cfg = OmegaConf.load(args.config)

    # Setup an experiment folder:
    if accelerator.is_local_main_process:
        exp_dir = cfg.train.exp_dir
        os.makedirs(exp_dir, exist_ok=True)
        ckpt_dir = os.path.join(exp_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        print(f"Experiment directory created at {exp_dir}")

    # Create model:
    cldm: ControlLDM = instantiate_from_config(cfg.model.cldm)
    sd = torch.load(cfg.train.sd_path, map_location="cpu")["state_dict"]
    unused = cldm.load_pretrained_sd(sd)
    if accelerator.is_local_main_process:
        print(f"strictly load pretrained SD weight from {cfg.train.sd_path}\n"
              f"unused weights: {unused}")
    
    if cfg.train.resume:
        cldm.load_controlnet_from_ckpt(torch.load(cfg.train.resume, map_location="cpu"))
        if accelerator.is_local_main_process:
            print(f"strictly load controlnet weight from checkpoint: {cfg.train.resume}")
    else:
        init_with_new_zero, init_with_scratch = cldm.load_controlnet_from_unet()
        if accelerator.is_local_main_process:
            print(f"strictly load controlnet weight from pretrained SD\n"
                  f"weights initialized with newly added zeros: {init_with_new_zero}\n"
                  f"weights initialized from scratch: {init_with_scratch}")
    
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    sd = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in torch.load(cfg.train.swinir_path, map_location="cpu").items()
    }
    swinir.load_state_dict(sd, strict=True)
    for p in swinir.parameters():
        p.requires_grad = False
    if accelerator.is_local_main_process:
        print(f"load SwinIR from {cfg.train.swinir_path}")
    
    diffusion: Diffusion = instantiate_from_config(cfg.model.diffusion)
    
    # Setup optimizer:
    opt = torch.optim.AdamW(cldm.controlnet.parameters(), lr=cfg.train.learning_rate)
    
    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True, drop_last=True
    )
    if accelerator.is_local_main_process:
        print(f"Dataset contains {len(dataset):,} images from {dataset.file_list}")

    # Prepare models for training:
    cldm.train().to(device)
    swinir.eval().to(device)
    diffusion.to(device)
    cldm, opt, loader = accelerator.prepare(cldm, opt, loader)
    pure_cldm: ControlLDM = accelerator.unwrap_model(cldm)
    
    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    epoch_loss = []
    sampler = SpacedSampler(diffusion.betas)
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")
    
    while global_step < max_steps:
        pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch", total=len(loader))
        for gt, lq, prompt in loader:
            gt = rearrange(gt, "b h w c -> b c h w").contiguous().float().to(device)
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(device)
            with torch.no_grad():
                z_0 = pure_cldm.vae_encode(gt)
                clean = swinir(lq)
                cond = pure_cldm.prepare_condition(clean, prompt)
            t = torch.randint(0, diffusion.num_timesteps, (z_0.shape[0],), device=device)
            
            loss = diffusion.p_losses(cldm, z_0, t, cond)
            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()

            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            epoch_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}")

            # Log loss values:
            if global_step % cfg.train.log_every == 0 and global_step > 0:
                # Gather values from all processes
                avg_loss = accelerator.gather(torch.tensor(step_loss, device=device).unsqueeze(0)).mean().item()
                step_loss.clear()
                if accelerator.is_local_main_process:
                    writer.add_scalar("loss/loss_simple_step", avg_loss, global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0 and global_step > 0:
                if accelerator.is_local_main_process:
                    checkpoint = pure_cldm.controlnet.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)

            if global_step % cfg.train.image_every == 0 or global_step == 1:
                N = 12
                log_clean = clean[:N]
                log_cond = {k:v[:N] for k, v in cond.items()}
                log_gt, log_lq = gt[:N], lq[:N]
                log_prompt = prompt[:N]
                cldm.eval()
                with torch.no_grad():
                    z = sampler.sample(
                        model=cldm, device=device, steps=50, batch_size=len(log_gt), x_size=z_0.shape[1:],
                        cond=log_cond, uncond=None, cfg_scale=1.0, x_T=None,
                        progress=accelerator.is_local_main_process, progress_leave=False
                    )
                    if accelerator.is_local_main_process:
                        for tag, image in [
                            ("image/samples", (pure_cldm.vae_decode(z) + 1) / 2),
                            ("image/gt", (log_gt + 1) / 2),
                            ("image/lq", log_lq),
                            ("image/condition", log_clean),
                            ("image/condition_decoded", (pure_cldm.vae_decode(log_cond["c_img"]) + 1) / 2),
                            ("image/prompt", (log_txt_as_img((512, 512), log_prompt) + 1) / 2)
                        ]:
                            writer.add_image(tag, make_grid(image, nrow=4), global_step)
                cldm.train()
            accelerator.wait_for_everyone()
            if global_step == max_steps:
                break
        
        pbar.close()
        epoch += 1
        avg_epoch_loss = accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0)).mean().item()
        epoch_loss.clear()
        if accelerator.is_local_main_process:
            writer.add_scalar("loss/loss_simple_epoch", avg_epoch_loss, global_step)

    if accelerator.is_local_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
