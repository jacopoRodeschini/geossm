# State-space models submodule


from .statespace import StateSpaceModel
from .statespace_results import StateSpaceResults

# this function is needed to be imported here so that it can be used in the LRSSM class,
# which is defined in the stmodel submodule
from .statespace import _filter_kernelJAX, _itype_for, _ensure_x64_for_dtype


__all__ = ["StateSpaceModel", "StateSpaceResults", "_filter_kernelJAX", "_itype_for", "_ensure_x64_for_dtype"]
