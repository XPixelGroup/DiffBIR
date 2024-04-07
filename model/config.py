from typing import Optional, Literal
from types import ModuleType
import enum
from packaging import version

import torch

# collect system information
if version.parse(torch.__version__) >= version.parse("2.0.0"):
    SDP_IS_AVAILABLE = True
else:
    SDP_IS_AVAILABLE = False

try:
    import xformers
    import xformers.ops
    XFORMERS_IS_AVAILBLE = True
except:
    XFORMERS_IS_AVAILBLE = False


class AttnMode(enum.Enum):
    SDP = 0
    XFORMERS = 1
    VANILLA = 2


class Config:
    xformers: Optional[ModuleType] = None
    attn_mode: AttnMode = AttnMode.VANILLA


# initialize attention mode
if SDP_IS_AVAILABLE:
    Config.attn_mode = AttnMode.SDP
    print(f"use sdp attention as default")
elif XFORMERS_IS_AVAILBLE:
    Config.attn_mode = AttnMode.XFORMERS
    print(f"use xformers attention as default")
else:
    print(f"both sdp attention and xformers are not available, use vanilla attention (very expensive) as default")

if XFORMERS_IS_AVAILBLE:
    Config.xformers = xformers


def set_attn_mode(attn_mode: Literal["vanilla", "sdp", "xformers"]) -> None:
    assert attn_mode in ["vanilla", "sdp", "xformers"]
    if attn_mode == "sdp":
        assert SDP_IS_AVAILABLE
        Config.attn_mode = AttnMode.SDP
    elif attn_mode == "xformers":
        assert XFORMERS_IS_AVAILBLE
        Config.attn_mode = AttnMode.XFORMERS
    else:
        Config.attn_mode = AttnMode.VANILLA
    print(f"set attn_mode to {attn_mode}")
