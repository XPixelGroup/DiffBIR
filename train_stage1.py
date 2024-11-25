import os
from argparse import ArgumentParser
import warnings

from omegaconf import OmegaConf
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
from accelerate import Accelerator
from accelerate.utils import set_seed
from einops import rearrange
from tqdm import tqdm
import lpips

from diffbir.model import SwinIR
from diffbir.utils.common import instantiate_from_config, calculate_psnr_pt, to


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
    swinir: SwinIR = instantiate_from_config(cfg.model.swinir)
    if cfg.train.resume:
        swinir.load_state_dict(
            torch.load(cfg.train.resume, map_location="cpu"), strict=True
        )
        if accelerator.is_local_main_process:
            print(f"strictly load weight from checkpoint: {cfg.train.resume}")
    else:
        if accelerator.is_local_main_process:
            print("initialize from scratch")

    # Setup optimizer:
    opt = torch.optim.AdamW(
        swinir.parameters(), lr=cfg.train.learning_rate, weight_decay=0
    )

    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True,
        drop_last=True,
    )
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False,
        drop_last=False,
    )
    if accelerator.is_local_main_process:
        print(f"Dataset contains {len(dataset):,} images from {dataset.file_list}")

    batch_transform = instantiate_from_config(cfg.batch_transform)

    # Prepare models for training:
    swinir.train().to(device)
    swinir, opt, loader, val_loader = accelerator.prepare(
        swinir, opt, loader, val_loader
    )
    pure_swinir = accelerator.unwrap_model(swinir)

    # Variables for monitoring/logging purposes:
    global_step = 0
    max_steps = cfg.train.train_steps
    step_loss = []
    epoch = 0
    epoch_loss = []
    with warnings.catch_warnings():
        # avoid warnings from lpips internal
        warnings.simplefilter("ignore")
        lpips_model = (
            lpips.LPIPS(net="alex", verbose=accelerator.is_local_main_process)
            .eval()
            .to(device)
        )
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")

    while global_step < max_steps:
        pbar = tqdm(
            iterable=None,
            disable=not accelerator.is_local_main_process,
            unit="batch",
            total=len(loader),
        )
        for batch in loader:
            to(batch, device)
            batch = batch_transform(batch)
            gt, lq, _ = batch
            gt = rearrange((gt + 1) / 2, "b h w c -> b c h w").contiguous().float()
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float()
            pred = swinir(lq)
            loss = F.mse_loss(pred, gt, reduction="sum")

            opt.zero_grad()
            accelerator.backward(loss)
            opt.step()
            accelerator.wait_for_everyone()

            global_step += 1
            step_loss.append(loss.item())
            epoch_loss.append(loss.item())
            pbar.update(1)
            pbar.set_description(
                f"Epoch: {epoch:04d}, Global Step: {global_step:07d}, Loss: {loss.item():.6f}"
            )

            # Log loss values:
            if global_step % cfg.train.log_every == 0:
                # Gather values from all processes
                avg_loss = (
                    accelerator.gather(
                        torch.tensor(step_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                step_loss.clear()
                if accelerator.is_local_main_process:
                    writer.add_scalar("train/loss_step", avg_loss, global_step)

            # Save checkpoint:
            if global_step % cfg.train.ckpt_every == 0:
                if accelerator.is_local_main_process:
                    checkpoint = pure_swinir.state_dict()
                    ckpt_path = f"{ckpt_dir}/{global_step:07d}.pt"
                    torch.save(checkpoint, ckpt_path)

            if global_step % cfg.train.image_every == 0 or global_step == 1:
                swinir.eval()
                N = 12
                log_gt, log_lq = gt[:N], lq[:N]
                with torch.no_grad():
                    log_pred = swinir(log_lq)
                if accelerator.is_local_main_process:
                    for tag, image in [
                        ("image/pred", log_pred),
                        ("image/gt", log_gt),
                        ("image/lq", log_lq),
                    ]:
                        writer.add_image(tag, make_grid(image, nrow=4), global_step)
                swinir.train()

            # Evaluate model:
            if global_step % cfg.train.val_every == 0:
                swinir.eval()
                val_loss = []
                val_lpips = []
                val_psnr = []
                val_pbar = tqdm(
                    iterable=None,
                    disable=not accelerator.is_local_main_process,
                    unit="batch",
                    total=len(val_loader),
                    leave=False,
                    desc="Validation",
                )
                for val_batch in val_loader:
                    to(val_batch, device)
                    val_batch = batch_transform(val_batch)
                    val_gt, val_lq, _ = val_batch
                    val_gt = (
                        rearrange((val_gt + 1) / 2, "b h w c -> b c h w")
                        .contiguous()
                        .float()
                    )
                    val_lq = (
                        rearrange(val_lq, "b h w c -> b c h w").contiguous().float()
                    )
                    with torch.no_grad():
                        val_pred = swinir(val_lq)
                        val_loss.append(
                            F.mse_loss(val_pred, val_gt, reduction="sum").item()
                        )
                        val_lpips.append(
                            lpips_model(val_pred, val_gt, normalize=True).mean().item()
                        )
                        val_psnr.append(
                            calculate_psnr_pt(val_pred, val_gt, crop_border=0)
                            .mean()
                            .item()
                        )
                    val_pbar.update(1)
                val_pbar.close()
                avg_val_loss = (
                    accelerator.gather(
                        torch.tensor(val_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                avg_val_lpips = (
                    accelerator.gather(
                        torch.tensor(val_lpips, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                avg_val_psnr = (
                    accelerator.gather(
                        torch.tensor(val_psnr, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                if accelerator.is_local_main_process:
                    for tag, val in [
                        ("val/loss", avg_val_loss),
                        ("val/lpips", avg_val_lpips),
                        ("val/psnr", avg_val_psnr),
                    ]:
                        writer.add_scalar(tag, val, global_step)
                swinir.train()

            accelerator.wait_for_everyone()

            if global_step == max_steps:
                break

        pbar.close()
        epoch += 1
        avg_epoch_loss = (
            accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0))
            .mean()
            .item()
        )
        epoch_loss.clear()
        if accelerator.is_local_main_process:
            writer.add_scalar("train/loss_epoch", avg_epoch_loss, global_step)

    if accelerator.is_local_main_process:
        print("done!")
        writer.close()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args)
