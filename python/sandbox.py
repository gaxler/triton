from audioop import add
from triton.code_gen import JITFunction

import triton
import triton.language as tl

from aot.compiler import compile, TritonCompileConfig

@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector
    y_ptr,  # *Pointer* to second input vector
    output_ptr,  # *Pointer* to output vector
    n_elements,  # Size of the vector
    BLOCK_SIZE   # Number of elements each program should process
                 # NOTE: `constexpr` so it can be used as a shape value
):
    # There are multiple 'program's processing different data. We identify which program
    # we are here
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0
    # This program will process inputs that are offset from the initial data.
    # for instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back to DRAM
    tl.store(output_ptr + offsets, output, mask=mask)



if __name__ == "__main__":
    
    scope = {"A": [16, 24], "K": [1, 8], "M": [1, 8], "N": [1, 8]}
    conf = TritonCompileConfig(device_idx=0)
    compile(add_kernel, ['f32', 'f32', 'f32'], ['i64'], ['A'], scope=scope, conf=conf, BLOCK_SIZE=1024)
    pass