from triton.code_gen import JITFunction
from aot.c_codegen import make_kernel_signature


def test_function_signature_generation():
    print(make_kernel_signature(["f32","f32","f32"], ["i32", "i32", "i32"], ["x", "y", "z", "n", "m", "k"]))
    return

test_function_signature_generation()
