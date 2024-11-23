"""
All models used in inference:
- DiffBIR-v1
  All tasks share the same pre-trained stable diffusion v2.1 (sd_v2.1).
-- BSR task
    stage-1 model (swinir_general): SwinIR trained on ImageNet-1k with Real-ESRGAN degradation.
    stage-2 model (v1_general): IRControlNet trained on ImageNet-1k.
-- BFR task
    stage-1 model (swinir_face): SwinIR pre-trained on FFHQ, borrowed from DifFace (https://github.com/zsyOAOA/DifFace.git)
    stage-2 model (v1_face): IRControlNet trained on FFHQ.
-- BID task
    The same as BSR task.

- DiffBIR-v2
  All tasks share the same pre-trained stable diffusion v2.1 (sd_v2.1).
  All tasks share the same stage-2 model (v2).
-- BSR task
    stage-1 model (bsrnet): BSRNet borrowed from BSRGAN (https://github.com/cszn/BSRGAN.git).
-- BFR task
    stage-1 model (swinir_face): SwinIR pre-trained on FFHQ, borrowed from DifFace (https://github.com/zsyOAOA/DifFace.git)
-- BID task
    stage-1 model (scunet_psnr): SCUNet-PSNR borrowed from SCUNet (https://github.com/cszn/SCUNet.git)

- DiffBIR-v2.1
  All tasks share the same pre-trained stable diffusion v2.1-zsnr (sd_v2.1_zsnr).
  All tasks share the same stage-2 model (v2.1).
-- BSR task
    stage-1 model (swinir_realesrgan): SwinIR trained on ImageNet-1k with Real-ESRGAN degradation.
-- BFR task
    stage-1 model (swinir_face): SwinIR pre-trained on FFHQ, borrowed from DifFace (https://github.com/zsyOAOA/DifFace.git)
-- BID task
    The same as BSR task.
"""
MODELS = {
    # --------------- stage-1 model weights ---------------
    "bsrnet": "https://github.com/cszn/KAIR/releases/download/v1.0/BSRNet.pth",
    # the following checkpoint is up-to-date, we use the old version in our paper
    # "swinir_face": "https://github.com/zsyOAOA/DifFace/releases/download/V1.0/General_Face_ffhq512.pth",
    "swinir_face": "https://huggingface.co/lxq007/DiffBIR/resolve/main/face_swinir_v1.ckpt",
    "scunet_psnr": "https://github.com/cszn/KAIR/releases/download/v1.0/scunet_color_real_psnr.pth",
    "swinir_general": "https://huggingface.co/lxq007/DiffBIR/resolve/main/general_swinir_v1.ckpt",
    "swinir_realesrgan": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/realesrgan_s4_swinir_100k.pth",
    # --------------- pre-trained stable diffusion weights ---------------
    "sd_v2.1": "https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt",
    "sd_v2.1_zsnr": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/sd2.1-base-zsnr-laionaes5.ckpt",
    # --------------- IRControlNet weights ---------------
    "v1_face": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_face.pth",
    "v1_general": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v1_general.pth",
    "v2": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/v2.pth",
    "v2.1": "https://huggingface.co/lxq007/DiffBIR-v2/resolve/main/DiffBIR_v2.1.pt",
}
