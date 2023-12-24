

class OptimizationFlag:
    xformers_attn = True
    sdp_attn = False
    autocast = False
    fp16 = False

def enable_autocast() -> None:
    OptimizationFlag.autocast = True

def check_optimization_flag() -> None:
    assert not (OptimizationFlag.autocast and OptimizationFlag.fp16)
