from typing import Dict, Sequence
import math
import random
import time

import numpy as np
import torch
from torch.utils import data
from PIL import Image

from utils.degradation import circular_lowpass_kernel, random_mixed_kernels
from utils.image import augment, random_crop_arr, center_crop_arr
from utils.file import load_file_list


class RealESRGANDataset(data.Dataset):
    """
    # TODO: add comment
    """

    def __init__(
        self,
        file_list: str,
        out_size: int,
        crop_type: str,
        use_hflip: bool,
        use_rot: bool,
        # blur kernel settings of the first degradation stage
        blur_kernel_size: int,
        kernel_list: Sequence[str],
        kernel_prob: Sequence[float],
        blur_sigma: Sequence[float],
        betag_range: Sequence[float],
        betap_range: Sequence[float],
        sinc_prob: float,
        # blur kernel settings of the second degradation stage
        blur_kernel_size2: int,
        kernel_list2: Sequence[str],
        kernel_prob2: Sequence[float],
        blur_sigma2: Sequence[float],
        betag_range2: Sequence[float],
        betap_range2: Sequence[float],
        sinc_prob2: float,
        final_sinc_prob: float
    ) -> "RealESRGANDataset":
        super(RealESRGANDataset, self).__init__()
        self.paths = load_file_list(file_list)
        self.out_size = out_size
        self.crop_type = crop_type
        assert self.crop_type in ["center", "random", "none"], f"invalid crop type: {self.crop_type}"

        self.blur_kernel_size = blur_kernel_size
        self.kernel_list = kernel_list
        # a list for each kernel probability
        self.kernel_prob = kernel_prob
        self.blur_sigma = blur_sigma
        # betag used in generalized Gaussian blur kernels
        self.betag_range = betag_range
        # betap used in plateau blur kernels
        self.betap_range = betap_range
        # the probability for sinc filters
        self.sinc_prob = sinc_prob

        self.blur_kernel_size2 = blur_kernel_size2
        self.kernel_list2 = kernel_list2
        self.kernel_prob2 = kernel_prob2
        self.blur_sigma2 = blur_sigma2
        self.betag_range2 = betag_range2
        self.betap_range2 = betap_range2
        self.sinc_prob2 = sinc_prob2
        
        # a final sinc filter
        self.final_sinc_prob = final_sinc_prob
        
        self.use_hflip = use_hflip
        self.use_rot = use_rot
        
        # kernel size ranges from 7 to 21
        self.kernel_range = [2 * v + 1 for v in range(3, 11)]
        # TODO: kernel range is now hard-coded, should be in the configure file
        # convolving with pulse tensor brings no blurry effect
        self.pulse_tensor = torch.zeros(21, 21).float()
        self.pulse_tensor[10, 10] = 1

    @torch.no_grad()
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        # -------------------------------- Load hq images -------------------------------- #
        hq_path = self.paths[index]
        success = False
        for _ in range(3):
            try:
                pil_img = Image.open(hq_path).convert("RGB")
                success = True
                break
            except:
                time.sleep(1)
        assert success, f"failed to load image {hq_path}"
        
        if self.crop_type == "random":
            pil_img = random_crop_arr(pil_img, self.out_size)
        elif self.crop_type == "center":
            pil_img = center_crop_arr(pil_img, self.out_size)
        # self.crop_type is "none"
        else:
            pil_img = np.array(pil_img)
            assert pil_img.shape[:2] == (self.out_size, self.out_size)
        # hwc, rgb to bgr, [0, 255] to [0, 1], float32
        img_hq = (pil_img[..., ::-1] / 255.0).astype(np.float32)

        # -------------------- Do augmentation for training: flip, rotation -------------------- #
        img_hq = augment(img_hq, self.use_hflip, self.use_rot)
        
        # ------------------------ Generate kernels (used in the first degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob:
            # this sinc filter setting is for kernels ranging from [7, 21]
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel = random_mixed_kernels(
                self.kernel_list,
                self.kernel_prob,
                kernel_size,
                self.blur_sigma,
                self.blur_sigma, [-math.pi, math.pi],
                self.betag_range,
                self.betap_range,
                noise_range=None
            )
        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel = np.pad(kernel, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------ Generate kernels (used in the second degradation) ------------------------ #
        kernel_size = random.choice(self.kernel_range)
        if np.random.uniform() < self.sinc_prob2:
            if kernel_size < 13:
                omega_c = np.random.uniform(np.pi / 3, np.pi)
            else:
                omega_c = np.random.uniform(np.pi / 5, np.pi)
            kernel2 = circular_lowpass_kernel(omega_c, kernel_size, pad_to=False)
        else:
            kernel2 = random_mixed_kernels(
                self.kernel_list2,
                self.kernel_prob2,
                kernel_size,
                self.blur_sigma2,
                self.blur_sigma2, [-math.pi, math.pi],
                self.betag_range2,
                self.betap_range2,
                noise_range=None
            )

        # pad kernel
        pad_size = (21 - kernel_size) // 2
        kernel2 = np.pad(kernel2, ((pad_size, pad_size), (pad_size, pad_size)))

        # ------------------------------------- the final sinc kernel ------------------------------------- #
        if np.random.uniform() < self.final_sinc_prob:
            kernel_size = random.choice(self.kernel_range)
            omega_c = np.random.uniform(np.pi / 3, np.pi)
            sinc_kernel = circular_lowpass_kernel(omega_c, kernel_size, pad_to=21)
            sinc_kernel = torch.FloatTensor(sinc_kernel)
        else:
            sinc_kernel = self.pulse_tensor

        # [0, 1], BGR to RGB, HWC to CHW
        img_hq = torch.from_numpy(
            img_hq[..., ::-1].transpose(2, 0, 1).copy()
        ).float()
        kernel = torch.FloatTensor(kernel)
        kernel2 = torch.FloatTensor(kernel2)

        return {
            "hq": img_hq, "kernel1": kernel, "kernel2": kernel2,
            "sinc_kernel": sinc_kernel, "txt": ""
        }

    def __len__(self) -> int:
        return len(self.paths)
