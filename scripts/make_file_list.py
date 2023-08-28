import sys
sys.path.append(".")
import os
from argparse import ArgumentParser

from utils.file import list_image_files


parser = ArgumentParser()
parser.add_argument("--img_folder", type=str, required=True)
parser.add_argument("--val_size", type=int, required=True)
parser.add_argument("--save_folder", type=str, required=True)
parser.add_argument("--follow_links", action="store_true")
args = parser.parse_args()

files = list_image_files(
    args.img_folder, exts=(".jpg", ".png", ".jpeg"), follow_links=args.follow_links,
    log_progress=True, log_every_n_files=10000
)

print(f"find {len(files)} images in {args.img_folder}")
assert args.val_size < len(files)

val_files = files[:args.val_size]
train_files = files[args.val_size:]

os.makedirs(args.save_folder, exist_ok=True)

with open(os.path.join(args.save_folder, "train.list"), "w") as fp:
    for file_path in train_files:
        fp.write(f"{file_path}\n")

with open(os.path.join(args.save_folder, "val.list"), "w") as fp:
    for file_path in val_files:
        fp.write(f"{file_path}\n")
