#!/usr/bin/env python3

import os
import torch

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'

torch.set_grad_enabled(False)

from torch._inductor import config
config.fx_graph_cache = False
config.force_disable_caches = True
config.max_autotune_gemm_backends = "TRITON"
config.max_autotune_gemm_search_space = "EXHAUSTIVE"

def main():
    device = "cuda"
    m, k, n = 51936, 295120, 112
    dtype = torch.float16
    
    print(f"Testing mm({m}, {k}) x ({k}, {n}) with {dtype}")
    
    a = torch.randn(m, k, device=device, dtype=dtype)
    b = torch.randn(k, n, device=device, dtype=dtype)
    
    @torch.compile(mode="max-autotune")
    def mm_fn(x, y):
        return torch.mm(x, y)
    
    try:
        result = mm_fn(a, b)
        print(f"✓ Success! Result shape: {result.shape}")
    except Exception as e:
        print(f"✗ CUDA Error: {e}")
        raise

if __name__ == "__main__":
    main()
