import torch
import lpips

from .image import rgb2ycbcr_pt
from .common import frozen_module


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


class LPIPS:
    
    def __init__(self, net: str) -> None:
        self.model = lpips.LPIPS(net=net)
        frozen_module(self.model)
    
    @torch.no_grad()
    def __call__(self, img1: torch.Tensor, img2: torch.Tensor, normalize: bool) -> torch.Tensor:
        """
        Compute LPIPS.
        
        Args:
            img1 (torch.Tensor): The first image (NCHW, RGB, [-1, 1]). Specify `normalize` if input 
                image is range in [0, 1].
            img2 (torch.Tensor): The second image (NCHW, RGB, [-1, 1]). Specify `normalize` if input 
                image is range in [0, 1].
            normalize (bool): If specified, the input images will be normalized from [0, 1] to [-1, 1].
            
        Returns:
            lpips_values (torch.Tensor): The lpips scores of this batch.
        """
        return self.model(img1, img2, normalize=normalize)
    
    def to(self, device: str) -> "LPIPS":
        self.model.to(device)
        return self
