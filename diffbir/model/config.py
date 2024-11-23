import os
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
if XFORMERS_IS_AVAILBLE:
    Config.attn_mode = AttnMode.XFORMERS
    print(f"use xformers attention as default")
elif SDP_IS_AVAILABLE:
    Config.attn_mode = AttnMode.SDP
    print(f"use sdp attention as default")
else:
    print(f"both sdp attention and xformers are not available, use vanilla attention (very expensive) as default")

if XFORMERS_IS_AVAILBLE:
    Config.xformers = xformers


# user-specified attention mode
ATTN_MODE = os.environ.get("ATTN_MODE", None)
if ATTN_MODE is not None:
    assert ATTN_MODE in ["vanilla", "sdp", "xformers"]
    if ATTN_MODE == "sdp":
        assert SDP_IS_AVAILABLE
        Config.attn_mode = AttnMode.SDP
    elif ATTN_MODE == "xformers":
        assert XFORMERS_IS_AVAILBLE
        Config.attn_mode = AttnMode.XFORMERS
    else:
        Config.attn_mode = AttnMode.VANILLA
    print(f"set attention mode to {ATTN_MODE}")
else:
    print("keep default attention mode")
