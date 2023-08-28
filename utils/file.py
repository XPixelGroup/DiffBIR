import os
from typing import List, Tuple


def load_file_list(file_list_path: str) -> List[str]:
    files = []
    # each line in file list contains a path of an image
    with open(file_list_path, "r") as fin:
        for line in fin:
            path = line.strip()
            if path:
                files.append(path)
    return files


def list_image_files(
    img_dir: str,
    exts: Tuple[str]=(".jpg", ".png", ".jpeg"),
    follow_links: bool=False,
    log_progress: bool=False,
    log_every_n_files: int=10000,
    max_size: int=-1
) -> List[str]:
    files = []
    for dir_path, _, file_names in os.walk(img_dir, followlinks=follow_links):
        early_stop = False
        for file_name in file_names:
            if os.path.splitext(file_name)[1].lower() in exts:
                if max_size >= 0 and len(files) >= max_size:
                    early_stop = True
                    break
                files.append(os.path.join(dir_path, file_name))
                if log_progress and len(files) % log_every_n_files == 0:
                    print(f"find {len(files)} images in {img_dir}")
        if early_stop:
            break
    return files


def get_file_name_parts(file_path: str) -> Tuple[str, str, str]:
    parent_path, file_name = os.path.split(file_path)
    stem, ext = os.path.splitext(file_name)
    return parent_path, stem, ext
