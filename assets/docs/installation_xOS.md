# Linux
Please follow the primary README.md of this repo.

# Windows
Windows users may stumble when installing the package `triton`. 
You can choose to run on **CPU** without `xformers` and `triton` installed.

To use **CUDA**, please refer to [issue#24](https://github.com/XPixelGroup/DiffBIR/issues/24) to try solve the problem of `triton` installation.

# MacOS
<!-- Currenly only CPU device is supported to run DiffBIR on Apple Silicon since most GPU acceleration packages are compatible with CUDA only. 

We are still trying to support MPS device. Stay tuned for our progress! -->

You can try to set up according to the following steps to use CPU or MPS device.

1. Install **torch (Preview/Nighly version)**.

    ```bash
    # MPS acceleration is available on MacOS 12.3+
    pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cpu
    ```
    Check more details in [official document](https://pytorch.org/get-started/locally/).

2. Package `triton` and `xformers` is not needed since they work with CUDA. Remove the related packages. 

    Your requirements.txt should look like:
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

3. [Run the inference script](https://github.com/XPixelGroup/DiffBIR#general_image_inference) and specify `--device cpu` or `--device mps`. Using MPS can accelarate your inference.

    You can specify `--tiled` and related arguments to avoid OOM. 