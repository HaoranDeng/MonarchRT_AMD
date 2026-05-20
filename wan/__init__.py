from . import _flashinfer_device_props_compat

_flashinfer_device_props_compat.ensure_patched_torch_cuda_device_properties()

from . import configs, distributed, modules
from .image2video import WanI2V
from .text2video import WanT2V
