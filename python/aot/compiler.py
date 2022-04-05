from dataclasses import dataclass
from typing import Mapping, Sequence
import triton
from triton.code_gen import JITFunction, Kernel
import triton._C.libtriton.triton as _triton

from .abstract_values import AbstractValue
from .sig_annotation_dsl import sig_generator
from .sig_annotation_dsl import ConfigKeys

@dataclass
class TritonCompileConfig:
    device_idx: int
    num_warps: int = 4
    num_stages: int = 4
    force_nc_cache: bool = False

    @classmethod
    def from_dict(cls, d):
        return cls(**d)

def _to_python_ir(obj: AbstractValue):
    name = obj.tt_dtype
    if hasattr(obj, 'data_ptr'):
        return 'ptr', name
    return 'scalar', name

def _handle_meta_constants(func: JITFunction, meta: Mapping):
    return {func.arg_names.index(name): triton.language.constexpr(value) for name, value in meta.items() if isinstance(value, int)}



def compile(func: JITFunction, pointers: Sequence[str], attributes: Sequence[str], attr_sizes: Sequence[str], scope, conf: TritonCompileConfig, **meta):  
    name = func.__name__
    constants = _handle_meta_constants(func, meta)
    for wargs in sig_generator(pointers=pointers, attributes=attributes, attr_vars=attr_sizes, named_vars=scope):
        tensor_idxs = [i for i, arg in enumerate(wargs) if hasattr(arg, 'data_ptr')]
        for i, pos in enumerate(sorted(constants)):
            wargs.insert(pos + i, constants[pos])
        # attributes
        attributes = dict()
        for i, arg in enumerate(wargs):
            if i in func.do_not_specialize:
                continue
            if isinstance(arg, int):
                attributes[i] = Kernel.pow2_divisor(arg)
            elif i in tensor_idxs:
                addr = arg.data_ptr()
                # This line checks an actual address on GPU. Since this is an AoT compiler,
                # we simply skip it.
                # range_size = _triton.runtime.get_pointer_range_size(addr)
                range_size = arg.data_ptr()
                attributes[i] = min(Kernel.pow2_divisor(addr),
                                    Kernel.pow2_divisor(range_size))
        # transforms ints whose value is one into constants for just-in-time compilation
        constants = {i: arg for i, arg in enumerate(wargs) if isinstance(arg, int) and arg == 1 and i not in func.do_not_specialize}
        constants.update({i: arg.value for i, arg in enumerate(wargs) if isinstance(arg, triton.language.constexpr)})
        constants.update({i: None for i, arg in enumerate(wargs) if arg is None})
        arg_types = [_to_python_ir(arg) for i, arg in enumerate(wargs) if i not in constants]
        compile = dict(arg_types=arg_types, device=conf.device_idx, attributes=attributes, constants=constants, num_warps=conf.num_warps, num_stages=conf.num_stages)
        attrib_sizes = [Kernel.pow2_divisor(s.val) for s in wargs if isinstance(s, AbstractValue) and s.is_attr]
        bin_ = func._compile(**compile)
        pass
        
    return




