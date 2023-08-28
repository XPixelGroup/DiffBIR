from typing import overload, Any, Dict
import torch


class ImageLoggerMixin:

    @overload
    def log_images(self, batch: Any, **kwargs: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        ...
