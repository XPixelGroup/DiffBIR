import sys
sys.path.append(".")
from argparse import ArgumentParser
import os
from typing import Any

from omegaconf import OmegaConf
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import pytorch_lightning as pl

from utils.common import instantiate_from_config


def wrap_dataloader(data_loader: DataLoader) -> Any:
    while True:
        yield from data_loader


pl.seed_everything(231, workers=True)

parser = ArgumentParser()
parser.add_argument("--config", type=str, required=True)
parser.add_argument("--sample_size", type=int, default=128)
parser.add_argument("--show_gt", action="store_true")
parser.add_argument("--output", type=str, required=True)
args = parser.parse_args()

config = OmegaConf.load(args.config)
dataset = instantiate_from_config(config.dataset)
transform = instantiate_from_config(config.batch_transform)
data_loader = wrap_dataloader(DataLoader(dataset, batch_size=1, shuffle=True))

cnt = 0
os.makedirs(args.output, exist_ok=True)

for batch in data_loader:
    batch = transform(batch)
    for hq, lq in zip(batch["jpg"], batch["hint"]):
        hq = ((hq + 1) * 127.5).numpy().clip(0, 255).astype(np.uint8)
        lq = (lq * 255.0).numpy().clip(0, 255).astype(np.uint8)
        if args.show_gt:
            Image.fromarray(np.concatenate([hq, lq], axis=1)).save(os.path.join(args.output, f"{cnt}.png"))
        else:
            Image.fromarray(lq).save(os.path.join(args.output, f"{cnt}.png"))
        cnt += 1
        if cnt >= args.sample_size:
            break
    if cnt >= args.sample_size:
        break
