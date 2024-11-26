from typing import Mapping, Any, Tuple, Callable, Dict, Literal
import importlib
import os
from urllib.parse import urlparse

import torch
from torch import Tensor
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw, ImageFont
from torch.hub import download_url_to_file, get_dir


def get_obj_from_str(string: str, reload: bool = False) -> Any:
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)


def instantiate_from_config(config: Mapping[str, Any]) -> Any:
    if not "target" in config:
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))


def wavelet_blur(image: Tensor, radius: int):
    """
    Apply wavelet blur to the input tensor.
    """
    # input shape: (1, 3, H, W)
    # convolution kernel
    kernel_vals = [
        [0.0625, 0.125, 0.0625],
        [0.125, 0.25, 0.125],
        [0.0625, 0.125, 0.0625],
    ]
    kernel = torch.tensor(kernel_vals, dtype=image.dtype, device=image.device)
    # add channel dimensions to the kernel to make it a 4D tensor
    kernel = kernel[None, None]
    # repeat the kernel across all input channels
    kernel = kernel.repeat(3, 1, 1, 1)
    image = F.pad(image, (radius, radius, radius, radius), mode="replicate")
    # apply convolution
    output = F.conv2d(image, kernel, groups=3, dilation=radius)
    return output


def wavelet_decomposition(image: Tensor, levels=5):
    """
    Apply wavelet decomposition to the input tensor.
    This function only returns the low frequency & the high frequency.
    """
    high_freq = torch.zeros_like(image)
    for i in range(levels):
        radius = 2**i
        low_freq = wavelet_blur(image, radius)
        high_freq += image - low_freq
        image = low_freq

    return high_freq, low_freq


def wavelet_reconstruction(content_feat: Tensor, style_feat: Tensor):
    """
    Apply wavelet decomposition, so that the content will have the same color as the style.
    """
    # calculate the wavelet decomposition of the content feature
    content_high_freq, content_low_freq = wavelet_decomposition(content_feat)
    del content_low_freq
    # calculate the wavelet decomposition of the style feature
    style_high_freq, style_low_freq = wavelet_decomposition(style_feat)
    del style_high_freq
    # reconstruct the content feature with the style's high frequency
    return content_high_freq + style_low_freq


# https://github.com/XPixelGroup/BasicSR/blob/master/basicsr/utils/download_util.py/
def load_file_from_url(url, model_dir=None, progress=True, file_name=None):
    """Load file form http url, will download models if necessary.

    Ref:https://github.com/1adrianb/face-alignment/blob/master/face_alignment/utils.py

    Args:
        url (str): URL to be downloaded.
        model_dir (str): The path to save the downloaded model. Should be a full path. If None, use pytorch hub_dir.
            Default: None.
        progress (bool): Whether to show the download progress. Default: True.
        file_name (str): The downloaded file name. If None, use the file name in the url. Default: None.

    Returns:
        str: The path to the downloaded file.
    """
    if model_dir is None:  # use the pytorch hub_dir
        hub_dir = get_dir()
        model_dir = os.path.join(hub_dir, "checkpoints")

    os.makedirs(model_dir, exist_ok=True)

    parts = urlparse(url)
    filename = os.path.basename(parts.path)
    if file_name is not None:
        filename = file_name
    cached_file = os.path.abspath(os.path.join(model_dir, filename))
    if not os.path.exists(cached_file):
        print(f'Downloading: "{url}" to {cached_file}\n')
        download_url_to_file(url, cached_file, hash_prefix=None, progress=progress)
    return cached_file


def load_model_from_url(url: str) -> Dict[str, torch.Tensor]:
    sd_path = load_file_from_url(url, model_dir="weights")
    sd = torch.load(sd_path, map_location="cpu")
    if "state_dict" in sd:
        sd = sd["state_dict"]
    if list(sd.keys())[0].startswith("module"):
        sd = {k[len("module.") :]: v for k, v in sd.items()}
    return sd


def sliding_windows(
    h: int, w: int, tile_size: int, tile_stride: int
) -> Tuple[int, int, int, int]:
    hi_list = list(range(0, h - tile_size + 1, tile_stride))
    if (h - tile_size) % tile_stride != 0:
        hi_list.append(h - tile_size)

    wi_list = list(range(0, w - tile_size + 1, tile_stride))
    if (w - tile_size) % tile_stride != 0:
        wi_list.append(w - tile_size)

    coords = []
    for hi in hi_list:
        for wi in wi_list:
            coords.append((hi, hi + tile_size, wi, wi + tile_size))
    return coords


# https://github.com/csslc/CCSR/blob/main/model/q_sampler.py#L503
def gaussian_weights(tile_width: int, tile_height: int) -> np.ndarray:
    """Generates a gaussian mask of weights for tile contributions"""
    latent_width = tile_width
    latent_height = tile_height
    var = 0.01
    midpoint = (
        latent_width - 1
    ) / 2  # -1 because index goes from 0 to latent_width - 1
    x_probs = [
        np.exp(
            -(x - midpoint) * (x - midpoint) / (latent_width * latent_width) / (2 * var)
        )
        / np.sqrt(2 * np.pi * var)
        for x in range(latent_width)
    ]
    midpoint = latent_height / 2
    y_probs = [
        np.exp(
            -(y - midpoint)
            * (y - midpoint)
            / (latent_height * latent_height)
            / (2 * var)
        )
        / np.sqrt(2 * np.pi * var)
        for y in range(latent_height)
    ]
    weights = np.outer(y_probs, x_probs)
    return weights


def make_tiled_fn(
    fn: Callable[[torch.Tensor], torch.Tensor],
    size: int,
    stride: int,
    scale_type: Literal["up", "down"] = "up",
    scale: int = 1,
    channel: int | None = None,
    weight: Literal["uniform", "gaussian"] = "gaussian",
    dtype: torch.dtype | None = None,
    device: torch.device | None = None,
    # callback: Callable[[int, int, int, int], None] | None = None,
    progress: bool = True,
) -> Callable[[torch.Tensor], torch.Tensor]:
    # Only split the first input of function.
    def tiled_fn(x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        if scale_type == "up":
            scale_fn = lambda n: int(n * scale)
        else:
            scale_fn = lambda n: int(n // scale)

        b, c, h, w = x.size()
        out_dtype = dtype or x.dtype
        out_device = device or x.device
        out_channel = channel or c
        out = torch.zeros(
            (b, out_channel, scale_fn(h), scale_fn(w)),
            dtype=out_dtype,
            device=out_device,
        )
        count = torch.zeros_like(out, dtype=torch.float32)
        weight_size = scale_fn(size)
        weights = (
            gaussian_weights(weight_size, weight_size)[None, None]
            if weight == "gaussian"
            else np.ones((1, 1, weight_size, weight_size))
        )
        weights = torch.tensor(
            weights,
            dtype=out_dtype,
            device=out_device,
        )

        indices = sliding_windows(h, w, size, stride)
        pbar = tqdm(
            indices, desc=f"Tiled Processing", disable=not progress, leave=False
        )
        for hi, hi_end, wi, wi_end in pbar:
            x_tile = x[..., hi:hi_end, wi:wi_end]
            out_hi, out_hi_end, out_wi, out_wi_end = map(
                scale_fn, (hi, hi_end, wi, wi_end)
            )
            if len(args) or len(kwargs):
                kwargs.update(dict(hi=hi, hi_end=hi_end, wi=wi, wi_end=wi_end))
            out[..., out_hi:out_hi_end, out_wi:out_wi_end] += (
                fn(x_tile, *args, **kwargs) * weights
            )
            count[..., out_hi:out_hi_end, out_wi:out_wi_end] += weights
        out = out / count
        return out

    return tiled_fn


TRACE_VRAM = int(os.environ.get("TRACE_VRAM", False))


def trace_vram_usage(tag: str) -> Callable:
    def wrapper_1(func: Callable) -> Callable:
        if not TRACE_VRAM:
            return func

        def wrapper_2(*args, **kwargs):
            peak_before = torch.cuda.max_memory_allocated() / (1024**3)
            ret = func(*args, **kwargs)
            torch.cuda.synchronize()
            peak_after = torch.cuda.max_memory_allocated() / (1024**3)
            YELLOW = "\033[93m"
            RESET = "\033[0m"
            print(
                f"{YELLOW}VRAM peak before {tag}: {peak_before:.5f} GB, "
                f"after: {peak_after:.5f} GB{RESET}"
            )
            return ret

        return wrapper_2

    return wrapper_1


class VRAMPeakMonitor:

    def __init__(self, tag: str) -> None:
        self.tag = tag

    def __enter__(self):
        self.peak_before = torch.cuda.max_memory_allocated() / (1024**3)
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        torch.cuda.synchronize()
        peak_after = torch.cuda.max_memory_allocated() / (1024**3)
        YELLOW = "\033[93m"
        RESET = "\033[0m"
        if TRACE_VRAM:
            print(
                f"{YELLOW}VRAM peak before {self.tag}: {self.peak_before:.2f} GB, "
                f"after: {peak_after:.2f} GB{RESET}"
            )
        return False


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
        lines = "\n".join(
            xc[bi][start : start + nc] for start in range(0, len(xc[bi]), nc)
        )

        try:
            draw.text((0, 0), lines, fill="black", font=font)
        except UnicodeEncodeError:
            print("Cant encode string for logging. Skipping.")

        txt = np.array(txt).transpose(2, 0, 1) / 127.5 - 1.0
        txts.append(txt)
    txts = np.stack(txts)
    txts = torch.tensor(txts)
    return txts


def to(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, dict):
        return {k: to(v, device) for k, v in obj.items()}
    if isinstance(obj, tuple):
        return tuple(to(v, device) for v in obj)
    if isinstance(obj, list):
        return [to(v, device) for v in obj]
    return obj


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
        out_img = (
            torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
        )
    else:
        weight = torch.tensor(
            [
                [65.481, -37.797, 112.0],
                [128.553, -74.203, -93.786],
                [24.966, 112.0, -18.214],
            ]
        ).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = (
            torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias
        )

    out_img = out_img / 255.0
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

    assert (
        img.shape == img2.shape
    ), f"Image shapes are different: {img.shape}, {img2.shape}."

    if crop_border != 0:
        img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
        img2 = img2[:, :, crop_border:-crop_border, crop_border:-crop_border]

    if test_y_channel:
        img = rgb2ycbcr_pt(img, y_only=True)
        img2 = rgb2ycbcr_pt(img2, y_only=True)

    img = img.to(torch.float64)
    img2 = img2.to(torch.float64)

    mse = torch.mean((img - img2) ** 2, dim=[1, 2, 3])
    return 10.0 * torch.log10(1.0 / (mse + 1e-8))
