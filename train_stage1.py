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

from model import SwinIR
from utils.common import instantiate_from_config


# https://github.com/XPixelGroup/BasicSR/blob/033cd6896d898fdd3dcda32e3102a792efa1b8f4/basicsr/utils/color_util.py#L186
def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).

    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.

    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.

    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img


# https://github.com/XPixelGroup/BasicSR/blob/033cd6896d898fdd3dcda32e3102a792efa1b8f4/basicsr/metrics/psnr_ssim.py#L52
def calculate_psnr_pt(img, img2, crop_border, test_y_channel=False):
    """Calculate PSNR (Peak Signal-to-Noise Ratio) (PyTorch version).

    Reference: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        crop_border (int): Cropped pixels in each edge of an image. These pixels are not involved in the calculation.
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: PSNR result.
    """

    assert img.shape == img2.shape, (f'Image shapes are different: {img.shape}, {img2.shape}.')

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2)**2, dim=[1, 2, 3])
    return 10. * torch.log10(1. / (mse + 1e-8))


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
        swinir.load_state_dict(torch.load(cfg.train.resume, map_location="cpu"), strict=True)
        if accelerator.is_local_main_process:
            print(f"strictly load weight from checkpoint: {cfg.train.resume}")
    else:
        if accelerator.is_local_main_process:
            print("initialize from scratch")
    
    # Setup optimizer:
    opt = torch.optim.AdamW(
        swinir.parameters(), lr=cfg.train.learning_rate,
        weight_decay=0
    )
    
    # Setup data:
    dataset = instantiate_from_config(cfg.dataset.train)
    loader = DataLoader(
        dataset=dataset, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=True, drop_last=True
    )
    val_dataset = instantiate_from_config(cfg.dataset.val)
    val_loader = DataLoader(
        dataset=val_dataset, batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        shuffle=False, drop_last=False
    )
    if accelerator.is_local_main_process:
        print(f"Dataset contains {len(dataset):,} images from {dataset.file_list}")

    # Prepare models for training:
    swinir.train().to(device)
    swinir, opt, loader, val_loader = accelerator.prepare(swinir, opt, loader, val_loader)
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
        lpips_model = lpips.LPIPS(net="alex", verbose=accelerator.is_local_main_process).eval().to(device)
    if accelerator.is_local_main_process:
        writer = SummaryWriter(exp_dir)
        print(f"Training for {max_steps} steps...")
    
    while global_step < max_steps:
        pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch", total=len(loader))
        for gt, lq, _ in loader:
            gt = rearrange((gt + 1) / 2, "b h w c -> b c h w").contiguous().float().to(device)
            lq = rearrange(lq, "b h w c -> b c h w").contiguous().float().to(device)
            pred = swinir(lq)
            loss = F.mse_loss(input=pred, target=gt, reduction="sum")

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
            if global_step % cfg.train.log_every == 0:
                # Gather values from all processes
                avg_loss = accelerator.gather(torch.tensor(step_loss, device=device).unsqueeze(0)).mean().item()
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
                val_pbar = tqdm(iterable=None, disable=not accelerator.is_local_main_process, unit="batch",
                                total=len(val_loader), leave=False, desc="Validation")
                # TODO: use accelerator.gather_for_metrics for more precise metric calculation?
                for val_gt, val_lq, _ in val_loader:
                    val_gt = rearrange((val_gt + 1) / 2, "b h w c -> b c h w").contiguous().float().to(device)
                    val_lq = rearrange(val_lq, "b h w c -> b c h w").contiguous().float().to(device)
                    with torch.no_grad():
                        # forward
                        val_pred = swinir(val_lq)
                        # compute metrics (loss, lpips, psnr)
                        val_loss.append(F.mse_loss(input=val_pred, target=val_gt, reduction="sum").item())
                        val_lpips.append(lpips_model(val_pred, val_gt, normalize=True).mean().item())
                        val_psnr.append(calculate_psnr_pt(val_pred, val_gt, crop_border=0).mean().item())
                    val_pbar.update(1)
                val_pbar.close()
                avg_val_loss = accelerator.gather(torch.tensor(val_loss, device=device).unsqueeze(0)).mean().item()
                avg_val_lpips = accelerator.gather(torch.tensor(val_lpips, device=device).unsqueeze(0)).mean().item()
                avg_val_psnr = accelerator.gather(torch.tensor(val_psnr, device=device).unsqueeze(0)).mean().item()
                if accelerator.is_local_main_process:
                    for tag, val in [
                        ("val/loss", avg_val_loss),
                        ("val/lpips", avg_val_lpips),
                        ("val/psnr", avg_val_psnr)
                    ]:
                        writer.add_scalar(tag, val, global_step)
                swinir.train()
            
            accelerator.wait_for_everyone()

            if global_step == max_steps:
                break
        
        pbar.close()
        epoch += 1
        avg_epoch_loss = accelerator.gather(torch.tensor(epoch_loss, device=device).unsqueeze(0)).mean().item()
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
