import subprocess
import sys

libs = [
    "pytorch_tabnet",
    "rtdl_revisiting_models",
    "deepctr_torch",
    "torchfm",
]
for lib in libs:
    try:
        __import__(lib)
        print(f"  [OK]    {lib}")
    except ImportError as e:
        print(f"  [MISS]  {lib}: {e}")
