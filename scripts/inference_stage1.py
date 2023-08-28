import sys
sys.path.append(".")
import os
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
from omegaconf import OmegaConf
import torch
from PIL import Image
import numpy as np
from tqdm import tqdm

from utils.image import auto_resize, pad
from utils.common import load_state_dict, instantiate_from_config
from utils.file import list_image_files, get_file_name_parts


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--sr_scale", type=float, default=1)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--show_lq", action="store_true")
    parser.add_argument("--resize_back", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--skip_if_exist", action="store_true")
    parser.add_argument("--seed", type=int, default=231)
    return parser.parse_args()


@torch.no_grad()
def main():
    args = parse_args()
    pl.seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model: pl.LightningModule = instantiate_from_config(OmegaConf.load(args.config))
    load_state_dict(model, torch.load(args.ckpt, map_location="cpu"), strict=True)
    model.freeze()
    model.to(device)
    
    assert os.path.isdir(args.input)
    
    pbar = tqdm(list_image_files(args.input, follow_links=True))
    for file_path in pbar:
        pbar.set_description(file_path)
        save_path = os.path.join(args.output, os.path.relpath(file_path, args.input))
        parent_path, stem, _ = get_file_name_parts(save_path)
        save_path = os.path.join(parent_path, f"{stem}.png")
        if os.path.exists(save_path):
            if args.skip_if_exist:
                print(f"skip {save_path}")
                continue
            else:
                raise RuntimeError(f"{save_path} already exist")
        os.makedirs(parent_path, exist_ok=True)
        
        # load low-quality image and resize
        lq = Image.open(file_path).convert("RGB")
        if args.sr_scale != 1:
            lq = lq.resize(
                tuple(int(x * args.sr_scale) for x in lq.size), Image.BICUBIC
            )
        lq_resized = auto_resize(lq, args.image_size)
        # padding
        x = pad(np.array(lq_resized), scale=64)

        x = torch.tensor(x, dtype=torch.float32, device=device) / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0).contiguous()
        try:
            pred = model(x).detach().squeeze(0).permute(1, 2, 0) * 255
            pred = pred.clamp(0, 255).to(torch.uint8).cpu().numpy()
        except RuntimeError as e:
            print(f"inference failed, error: {e}")
            continue
        
        # remove padding
        pred = pred[:lq_resized.height, :lq_resized.width, :]
        if args.show_lq:
            if args.resize_back:
                lq = np.array(lq)
                if lq_resized.size != lq.size:
                    pred = np.array(Image.fromarray(pred).resize(lq.size, Image.LANCZOS))
            else:
                lq = np.array(lq_resized)
            final_image = Image.fromarray(np.concatenate([lq, pred], axis=1))
        else:
            if args.resize_back and lq_resized.size != lq.size:
                final_image = Image.fromarray(pred).resize(lq.size, Image.LANCZOS)
            else:
                final_image = Image.fromarray(pred)
        final_image.save(save_path)


if __name__ == "__main__":
    main()
