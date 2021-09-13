from genericpath import exists
from io import UnsupportedOperation
from typing import Any, Mapping, Optional, Sequence
from .compilation_config import sig_generator, TritonCompileConfig
from triton.code_gen import JITFunction
import tempfile
from pathlib import Path
from dataclasses import dataclass

from ._types import NamedVariantsMap
from .tt_bindings import AOTKernel, filter_jit_funcs

from .c_codegen import KernelDispatcher, KernelLibrarySource

@dataclass
class LibCudaConf:
    include_dir: str
    link_dir: str


def temp_compile(*file_names):

    return


@dataclass
class Compiler:
    attr_size_scope: NamedVariantsMap
    tt_compile_conf: TritonCompileConfig
    scope: Optional[Mapping[str, Any]] = None

    def __enter__(self):
        self._called_with_context = True
        self._tmp_dir = Path(tempfile.mkdtemp(prefix="tt_aot_compiler"))
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        import shutil
        shutil.rmtree(str(self._tmp_dir))
        return

    def __post_init__(self):
        self._called_with_context = False
        if self.scope is None:
            self.scope = {}

        self.scope = {v:k for k,v in filter_jit_funcs(self.scope).items()}

        self.lib = KernelLibrarySource()
        return

    def _get_func_name(self, func: JITFunction) -> str:
        if not self._called_with_context:
            raise ValueError("Compiler must be used as a context")
        
        kernel_name = self.scope.get(func)
        if kernel_name is None:
            # func must be in calling scope
            # TODO: can someone define a compiler context and call compile in totaly different context?
            raise ValueError("Compiler must be used as a context")
        
        return kernel_name

    def compile(self, output: str = None):
        if output is None:
            output = self._tmp_dir / "obj_files"
            output.mkdir(exists_ok=True)
        output = Path(output)
        
        return
    
    def c_code_gen(self, func: JITFunction, pointers: Sequence[str], attributes: Sequence[str], attr_sizes: Sequence[str], **meta) -> KernelDispatcher:
        
        kernel_name = self._get_func_name(func)
        
        kernel = AOTKernel(func)

        print(f"Compiling {kernel_name}...:")
        code_gen = KernelDispatcher(kernel_name)
        
        for sig in sig_generator(pointers, attributes, attr_sizes, self.attr_size_scope):
            _, bin_ = kernel.aot_compile(
                *sig,
                num_warps=self.tt_compile_conf.num_warps,
                num_stages=self.tt_compile_conf.num_stages,
                force_nc_cache=self.tt_compile_conf.force_nc_cache,
                **meta,
            )

            attr_vals = [AOTKernel.pow2_divisor(s.val) for s in sig if s.is_attr]
            ptx = bin_.asm('ptx')
            code_gen.add_kernel(ptx, attr_vals, sig, func.arg_names)
            

            sig_repr = " ".join([f"{n}:{s}" for n, s in zip(func.arg_names, sig)])
            print(f"\t-> Done: {sig_repr}")
        
        self.lib.add(code_gen)
