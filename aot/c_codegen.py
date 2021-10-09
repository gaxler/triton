from dataclasses import dataclass, field
import binascii
from pathlib import Path
from typing import Sequence, Tuple


from .abstract_values import AbstractPtr, AbstractValue


THREADS_PER_WARP = 32
KERNEL_CONFIG_DATASTRUCT_NAME = "g"


@dataclass
class CInputSigneture:
    """Build kernel specific function signiture"""

    # TODO: make sure the mapping is correct
    _TRITON_TO_C_TYPE = {
        "I": "int32_t",
        "f": "float",
        "B": "bool",
        "f8": "float",
        "f16": "float",
        "bf16": "float",
        "f32": "float",
        "f64": "double",
        "i1": "bool",
        "i8": "int8_t",
        "i16": "int16_t",
        "i32": "int32_t",
        "i64": "int64_t",
    }

    input_types: Sequence[AbstractValue]
    arg_names: Sequence[str]

    def __len__(self):
        return len(self.input_types)

    def arg_pointers(self) -> str:
        return ", ".join([f"&{arg}" for arg in self.arg_names])

    def signeture(self) -> str:
        sig = ""
        for arg, ty in zip(self.arg_names, self.input_types):
            if isinstance(ty, AbstractPtr):
                dtype = "CUdeviceptr"
            else:
                dtype = CInputSigneture._TRITON_TO_C_TYPE[ty.tt_dtype]
            sig = f"{sig}{dtype} {arg}, "
        if len(sig):
            sig = sig[:-2]
            return sig


def py_str_to_uchar_array(txt: str) -> Tuple[str, int]:  # (c_code, array len)
    """Hexdump as string into a C array"""
    hex_ = str(binascii.hexlify(bytes(txt, "utf-8")))[2:-1]
    it = iter(hex_)
    data = ", ".join([f"0x{x}{next(it)}" for x in it])
    return data, len(hex_)


@dataclass
class CSource:
    name: str

    @property
    def include(self):
        return f'#include "{self.name}.h"'

    @property
    def header(self):
        if hasattr(self, "_header"):
            return self._header
        return None

    @property
    def source(self):
        if hasattr(self, "_source"):
            return self._source
        return None


@dataclass
class CommonHeaderCSource(CSource):
    name: str = "common"

    def __post_init__(self):
        header = """
#pragma once

#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>
#include <cuda.h>

typedef struct
{
    int gX;
    int gY;
    int gZ;
    int numWarps;
} GridWarps;

"""

        self._header = header


@dataclass
class _SingleKernelCSource(CSource):
    # name: str
    triton_output: str
    attr_sizes: Sequence[int]
    input_types: Sequence[AbstractValue]
    arg_names: Sequence[str]

    def __post_init__(self):
        hex_, bin_size = py_str_to_uchar_array(self.triton_output)
        signeture = CInputSigneture(self.input_types, self.arg_names)
        data = {
            "name": self.name,
            "bin_size": bin_size,
            "kernel_signeture": signeture.signeture(),
            "hex_": hex_,
            "num_args": len(signeture),
            "arg_pointers": signeture.arg_pointers(),
            "threads_per_warp": THREADS_PER_WARP,
        }

        header = """
unsigned char {name}_ptx[{bin_size}];
CUfunction load_{name}(void);
CUresult {name}(CUstream stream, GridWarps g, {kernel_signeture});
"""

        source = """
unsigned char {name}_ptx[{bin_size}] = 
{{
    {hex_}
}};

CUfunction load_{name}(void)
{{
    CUmodule mod_ptr;
    CUfunction func;
    CUresult err;
    void *image = (void *)&{name}_ptx;
    err = cuModuleLoadData(&mod_ptr, image);
    if (err != 0) {{
        return NULL;
    }}
    err = cuModuleGetFunction(&func, mod_ptr, "{name}");
    if (err != 0) {{
        return NULL;
    }}
    return func;
}}

CUresult {name}(CUstream stream, GridWarps g, {kernel_signeture})
{{
    CUfunction func = load_{name}();
    void *args[{num_args}] = {{ {arg_pointers} }};
    return cuLaunchKernel(func, g.gX, g.gY, g.gZ, g.numWarps * {threads_per_warp}, 1, 1, 0, stream, args, NULL);
}}
        """

        self._header = header.format(**data)
        self._source = source.format(**data)

    def signeture(self):
        return CInputSigneture(self.input_types, self.arg_names).signeture()

    def dispatch_cond(self) -> str:
        attrs = [name for v, name in zip(self.input_types, self.arg_names) if v.is_attr]
        assert len(attrs) == len(
            self.attr_sizes
        ), f"Got {len(attrs)} attributes and {len(self.attr_sizes)} attribute sizes"

        return " & ".join([f"{k} % {v} == 0" for k, v in zip(attrs, self.attr_sizes)])


@dataclass
class KernelDispatcher(CSource):
    """
    Dispatcher handles multiple version of the kernel, each version is optimized for a specific size of the input attributes.
    """
    name: str

    def __post_init__(self):
        self._header = None
        self._source = None
        self.kernels = []

    def add_kernel(
        self,
        triton_output: str, # ptx/cubin etc. Whatever cuda is going to lauch
        attribute_values: Sequence[int], # attribute sizes  that the kernel was optimized for
        signeture: Sequence[AbstractValue], # input signeture of the kernel (types with attribute sizes etc)
        arg_names: Sequence[str],  # names of 
    ):
        new_ker_name = f"{self.name}_" + "_".join(map(str, attribute_values))
        ptx = triton_output.replace(self.name, new_ker_name)
        single_ker = _SingleKernelCSource(
            new_ker_name, ptx, attribute_values, signeture, arg_names
        )
        self.kernels.append(single_ker)

    def generate_code(self, common_headers: Sequence[CSource]):
        disp_h, disp_c = self.dispatcher()
        common_includes = "\n".join([h.include for h in common_headers])
        data = {
            "name": self.name,
            "common": common_includes,
            "kernel_headers": "\n\n".join([k.header for k in self.kernels]),
            "kernel_sources": "\n\n".join([k.source for k in self.kernels]),
            "dispatcher_header": disp_h,
            "dispatcher_source": disp_c,
        }

        header = "{common}\n{kernel_headers}\n{dispatcher_header}"
        source = """
#include "{name}.h"

{kernel_sources}

{dispatcher_source}
        """

        self._header = header.format(**data)
        self._source = source.format(**data)

    def dispatcher(self):
        """
        Generate Size Optimized Kernel Dispatcher C code. Code does the following:
            * Determine what are the input attribute sizes 
            * Launch the size optimized kernel
        """
        last_kernel = self.kernels[-1]
        dispatch_sign = f"CUresult {self.name}(CUstream stream, GridWarps g, {last_kernel.signeture()})"
        header = f"{dispatch_sign};"

        conds = [
            f"if ({k.dispatch_cond()}) {{\n\treturn {k.name}(stream, g, {', '.join(k.arg_names)});\n}}"
            for k in self.kernels[:-1]
        ]
        conds = "\n\t".join(conds)

        last_return = (
            f"{last_kernel.name}(stream, g, {', '.join(last_kernel.arg_names)});"
        )

        source = """
{dispatch_sign} {{
{conds}
    return {last_return};
}}
""".format(
            **locals()
        )

        return header, source


@dataclass
class KernelLibrarySource:
    """ 
    Collection of Kernel Modules and their common code.

    Kernels are built as dispatchers. Dispatchers read the input size and dispatch the computation to size optimized cuda kerenl.
    """
    default_output: str = None
    common_headers: Sequence[CSource] = field(init=False)
    kernel_dispatchers: Sequence[KernelDispatcher] = field(init=False)

    def __post_init__(self):
        # TODO: if allow adding, make sure names are unique
        self.common_headers = [CommonHeaderCSource()]
        self.kernel_dispatchers = []
        
        if self.default_output is not None:
            # create temp file with common headers
            self._dump(self.default_output, self.common_headers, [])

    def add(self, kernel: KernelDispatcher):
        kernel.generate_code(self.common_headers)
        # self.kernel_dispatchers.append(kernel)
        if self.default_output is not None:
            self._dump(self.default_output, [kernel], [kernel])

    def _dump(
        self,
        output: str,
        headers: Sequence[CSource],
        sources: Sequence[KernelDispatcher],
    ):
        dst = Path(output)

        for h in headers:
            hfile = dst / f"{h.name}.h"
            hfile.write_text(h.header)

        for src in sources:
            cfile = dst / f"{src.name}.c"
            cfile.write_text(src.source)

    def dump_last(self, output: str):
        return self._dump(
            output,
            self.common_headers + [self.kernel_dispatchers[-1]],
            [self.kernel_dispatchers[-1]],
        )

    def dump(self, output: str):
        return self._dump(
            output, self.common_headers + self.kernel_dispatchers, self.kernel_dispatchers
        )

