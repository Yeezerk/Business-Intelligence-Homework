"""Syntax check: compile each model file to bytecode without execution."""
import py_compile
import os

files = [
    "d:/商业智能/商业智能大作业/code/models/mlp.py",
    "d:/商业智能/商业智能大作业/code/models/tabnet.py",
    "d:/商业智能/商业智能大作业/code/models/ft_transformer.py",
    "d:/商业智能/商业智能大作业/code/models/deepfm.py",
    "d:/商业智能/商业智能大作业/code/config.py",
]
for f in files:
    try:
        py_compile.compile(f, doraise=True)
        print(f"  [OK]   {os.path.basename(f)}")
    except py_compile.PyCompileError as e:
        print(f"  [FAIL] {os.path.basename(f)}: {e}")
