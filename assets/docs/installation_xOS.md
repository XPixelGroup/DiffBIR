# Linux
Please follow the primary README.md of this repo.

# Windows
Windows users may stumble when installing the package `triton`. 
You can choose to run on **CPU** without `xformers` and `triton` installed.

To use **CUDA**, please refer to [issue#24](https://github.com/XPixelGroup/DiffBIR/issues/24) to try solve the problem of `triton` installation.

# MacOS
Currenly only CPU device is supported to run DiffBIR on Apple Silicon since most GPU acceleration packages are compatiable with CUDA only. 

We are still trying to support MPS device. Stay tuned for our progress!

You can try to set up according to the following steps.

1. Install **torch** according to the [official document](https://pytorch.org/get-started/locally/).

```bash
pip install torch torchvision
```

2. Package `triton` and `xformers` is not needed since they work with CUDA.

Remove torch & cuda related packages. Your requirements.txt looks like:
```bash
# requirements.txt
pytorch_lightning==1.4.2
einops
open-clip-torch
omegaconf
torchmetrics==0.6.0
opencv-python-headless
scipy
matplotlib
lpips
gradio
chardet
transformers
facexlib
```

```bash
pip install -r requirements.txt
```

3. Run the inference script using CPU. Ensure you've downloaded the model weights.
```bash
python inference.py \
--input inputs/demo/general \
--config configs/model/cldm.yaml \
--ckpt weights/general_full_v1.ckpt \
--reload_swinir --swinir_ckpt weights/general_swinir_v1.ckpt \
--steps 50 \
--sr_scale 4 \
--image_size 512 \
--color_fix_type wavelet --resize_back \
--output results/demo/general \
--device cpu
```