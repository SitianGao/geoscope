from collections import namedtuple
import os
from pathlib import Path
from string import Template
import torch


_CUDA_DLL_DIRECTORY = None
if os.name == "nt" and hasattr(os, "add_dll_directory"):
    torch_cuda_lib = Path(torch.__file__).resolve().parent / "lib"
    if torch_cuda_lib.is_dir():
        _CUDA_DLL_DIRECTORY = os.add_dll_directory(str(torch_cuda_lib))

import cupy


Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


@cupy.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_module = cupy.RawModule(code=code)
    return kernel_module.get_function(kernel_name)
