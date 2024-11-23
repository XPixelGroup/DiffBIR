from typing import Any, overload, Dict, Union, List, Sequence
import random

import torch
from torch.nn import functional as F
import numpy as np

from .utils import USMSharp, DiffJPEG, filter2D
from .degradation import (
    random_add_gaussian_noise_pt,
    random_add_poisson_noise_pt,
)


class BatchTransform:

    @overload
    def __call__(self, batch: Any) -> Any: ...


class IdentityBatchTransform(BatchTransform):

    def __call__(self, batch: Any) -> Any:
        return batch


class RealESRGANBatchTransform(BatchTransform):

    def __init__(
        self,
        use_sharpener: bool,
        queue_size: int,
        resize_prob: Sequence[float],
        resize_range: Sequence[float],
        gray_noise_prob: float,
        gaussian_noise_prob: float,
        noise_range: Sequence[float],
        poisson_scale_range: Sequence[float],
        jpeg_range: Sequence[int],
        second_blur_prob: float,
        stage2_scale: Union[float, Sequence[Union[float, int]]],
        resize_prob2: Sequence[float],
        resize_range2: Sequence[float],
        gray_noise_prob2: float,
        gaussian_noise_prob2: float,
        noise_range2: Sequence[float],
        poisson_scale_range2: Sequence[float],
        jpeg_range2: Sequence[int],
    ) -> "RealESRGANBatchTransform":
        super().__init__()
        # resize settings for the first degradation process
        self.resize_prob = resize_prob
        self.resize_range = resize_range

        # noise settings for the first degradation process
        self.gray_noise_prob = gray_noise_prob
        self.gaussian_noise_prob = gaussian_noise_prob
        self.noise_range = noise_range
        self.poisson_scale_range = poisson_scale_range
        self.jpeg_range = jpeg_range

        self.second_blur_prob = second_blur_prob
        self.stage2_scale = stage2_scale
        assert isinstance(stage2_scale, (float, int)) or (
            isinstance(stage2_scale, Sequence)
            and len(stage2_scale) == 2
            and all(isinstance(x, (float, int)) for x in stage2_scale)
        ), f"stage2_scale can not be {type(stage2_scale)}"

        # resize settings for the second degradation process
        self.resize_prob2 = resize_prob2
        self.resize_range2 = resize_range2

        # noise settings for the second degradation process
        self.gray_noise_prob2 = gray_noise_prob2
        self.gaussian_noise_prob2 = gaussian_noise_prob2
        self.noise_range2 = noise_range2
        self.poisson_scale_range2 = poisson_scale_range2
        self.jpeg_range2 = jpeg_range2

        self.use_sharpener = use_sharpener
        if self.use_sharpener:
            self.usm_sharpener = USMSharp()
        else:
            self.usm_sharpener = None
        self.queue_size = queue_size
        self.jpeger = DiffJPEG(differentiable=False)

    @torch.no_grad()
    def _dequeue_and_enqueue(self):
        """It is the training pair pool for increasing the diversity in a batch.

        Batch processing limits the diversity of synthetic degradations in a batch. For example, samples in a
        batch could not have different resize scaling factors. Therefore, we employ this training pair pool
        to increase the degradation diversity in a batch.
        """
        # initialize
        b, c, h, w = self.lq.size()
        if not hasattr(self, "queue_lr"):
            assert (
                self.queue_size % b == 0
            ), f"queue size {self.queue_size} should be divisible by batch size {b}"
            self.queue_lr = torch.zeros(self.queue_size, c, h, w).to(self.lq)
            _, c, h, w = self.gt.size()
            self.queue_gt = torch.zeros(self.queue_size, c, h, w).to(self.lq)
            self.queue_txt = [None] * self.queue_size
            self.queue_ptr = 0
        if self.queue_ptr == self.queue_size:  # the pool is full
            # do dequeue and enqueue
            # shuffle
            idx = torch.randperm(self.queue_size)
            self.queue_lr = self.queue_lr[idx]
            self.queue_gt = self.queue_gt[idx]
            queue_txt_tmp = [None] * self.queue_size
            for i, j in enumerate(idx):
                queue_txt_tmp[i] = self.queue_txt[j]
            self.queue_txt = queue_txt_tmp
            # get first b samples
            lq_dequeue = self.queue_lr[0:b, :, :, :].clone()
            gt_dequeue = self.queue_gt[0:b, :, :, :].clone()
            txt_dequeue = self.queue_txt[0:b]
            # update the queue
            self.queue_lr[0:b, :, :, :] = self.lq.clone()
            self.queue_gt[0:b, :, :, :] = self.gt.clone()
            for i in range(b):
                self.queue_txt[i] = self.txt[i]
            self.lq = lq_dequeue
            self.gt = gt_dequeue
            self.txt = txt_dequeue
        else:
            # only do enqueue
            self.queue_lr[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                self.lq.clone()
            )
            self.queue_gt[self.queue_ptr : self.queue_ptr + b, :, :, :] = (
                self.gt.clone()
            )
            for i in range(b):
                self.queue_txt[self.queue_ptr + i] = self.txt[i]
            self.queue_ptr = self.queue_ptr + b

    @torch.no_grad()
    def __call__(
        self, batch: Dict[str, Union[torch.Tensor, str]]
    ) -> Dict[str, Union[torch.Tensor, List[str]]]:
        # training data synthesis
        hq = batch["hq"]
        if self.use_sharpener:
            self.usm_sharpener.to(hq)
            hq = self.usm_sharpener(hq)
        self.jpeger.to(hq)

        kernel1 = batch["kernel1"]
        kernel2 = batch["kernel2"]
        sinc_kernel = batch["sinc_kernel"]

        ori_h, ori_w = hq.size()[2:4]

        # ----------------------- The first degradation process ----------------------- #
        # blur
        out = filter2D(hq, kernel1)
        # random resize
        updown_type = random.choices(["up", "down", "keep"], self.resize_prob)[0]
        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range[1])
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range[0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(out, scale_factor=scale, mode=mode)
        # add noise
        if np.random.uniform() < self.gaussian_noise_prob:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.noise_range,
                clip=True,
                rounds=False,
                gray_prob=self.gray_noise_prob,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range,
                gray_prob=self.gray_noise_prob,
                clip=True,
                rounds=False,
            )
        # JPEG compression
        jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range)
        # clamp to [0, 1], otherwise JPEGer will result in unpleasant artifacts
        out = torch.clamp(out, 0, 1)
        # print("before jpeger: ", out.is_contiguous(), out.shape)
        out = self.jpeger(out, quality=jpeg_p)
        # print("after jpeger: ", out.is_contiguous(), out.shape)

        # ----------------------- The second degradation process ----------------------- #
        # blur
        if np.random.uniform() < self.second_blur_prob:
            out = filter2D(out, kernel2)

        # select scale of second degradation stage
        if isinstance(self.stage2_scale, Sequence):
            min_scale, max_scale = self.stage2_scale
            stage2_scale = np.random.uniform(min_scale, max_scale)
        else:
            stage2_scale = self.stage2_scale
        stage2_h, stage2_w = int(ori_h / stage2_scale), int(ori_w / stage2_scale)
        # print(f"stage2 scale = {stage2_scale}")

        # random resize
        updown_type = random.choices(["up", "down", "keep"], self.resize_prob2)[0]
        if updown_type == "up":
            scale = np.random.uniform(1, self.resize_range2[1])
        elif updown_type == "down":
            scale = np.random.uniform(self.resize_range2[0], 1)
        else:
            scale = 1
        mode = random.choice(["area", "bilinear", "bicubic"])
        out = F.interpolate(
            out, size=(int(stage2_h * scale), int(stage2_w * scale)), mode=mode
        )
        # add noise
        if np.random.uniform() < self.gaussian_noise_prob2:
            out = random_add_gaussian_noise_pt(
                out,
                sigma_range=self.noise_range2,
                clip=True,
                rounds=False,
                gray_prob=self.gray_noise_prob2,
            )
        else:
            out = random_add_poisson_noise_pt(
                out,
                scale_range=self.poisson_scale_range2,
                gray_prob=self.gray_noise_prob2,
                clip=True,
                rounds=False,
            )

        # JPEG compression + the final sinc filter
        # We also need to resize images to desired sizes. We group [resize back + sinc filter] together
        # as one operation.
        # We consider two orders:
        #   1. [resize back + sinc filter] + JPEG compression
        #   2. JPEG compression + [resize back + sinc filter]
        # Empirically, we find other combinations (sinc + JPEG + Resize) will introduce twisted lines.
        if np.random.uniform() < 0.5:
            # resize back + the final sinc filter
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
            out = filter2D(out, sinc_kernel)
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
        else:
            # JPEG compression
            jpeg_p = out.new_zeros(out.size(0)).uniform_(*self.jpeg_range2)
            out = torch.clamp(out, 0, 1)
            out = self.jpeger(out, quality=jpeg_p)
            # resize back + the final sinc filter
            mode = random.choice(["area", "bilinear", "bicubic"])
            out = F.interpolate(out, size=(stage2_h, stage2_w), mode=mode)
            out = filter2D(out, sinc_kernel)

        # resize back to gt_size since We are doing restoration task
        if stage2_scale != 1:
            out = F.interpolate(out, size=(ori_h, ori_w), mode="bicubic")
        # clamp and round
        lq = torch.clamp((out * 255.0).round(), 0, 255) / 255.0

        self.gt = hq
        self.lq = lq
        self.txt = batch["txt"]
        self._dequeue_and_enqueue()

        # [0, 1], float32, rgb, nhwc
        lq = self.lq.float().permute(0, 2, 3, 1).contiguous()
        # [-1, 1], float32, rgb, nhwc
        hq = (self.gt * 2 - 1).float().permute(0, 2, 3, 1).contiguous()
        txt = self.txt

        # return dict(jpg=hq, hint=lq, txt=batch["txt"])
        return hq, lq, txt
