import torch
import triton
import pytest

@pytest.mark.parametrize("M, N, dtype, mode",
    [
    (M, N, dtype, mode) for M in [1024, 821]
                        for N in [512, 857, 1871, 2089, 8573, 31000]
                        for dtype in ['float16', 'float32']\
                        for mode  in ['forward', 'backward']
    ]
                         )
def test_op(M, N, dtype, mode):
    dtype = {'float16': torch.float16, 'float32': torch.float32}[dtype]
    # create inputs
    x = torch.randn(M, N, dtype=dtype, device='cuda', requires_grad=True)
    idx = 4 + torch.ones(M, dtype=torch.int64, device='cuda')
    # forward pass
    tt_y = triton.ops.cross_entropy(x, idx)
    th_y = torch.nn.CrossEntropyLoss(reduction="none")(x, idx)
    if mode == 'forward':
        triton.testing.assert_almost_equal(th_y, tt_y)
    # backward pass
    elif mode == 'backward':
        dy = torch.randn_like(tt_y)
        # triton backward
        tt_y.backward(dy)
        tt_dx = x.grad.clone()
        # torch backward
        x.grad.zero_()
        th_y.backward(dy)
        th_dx = x.grad.clone()
        triton.testing.assert_almost_equal(th_dx, tt_dx)