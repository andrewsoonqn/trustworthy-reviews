import torch
import time

if torch.cuda.is_available():
    x = torch.randn(10000, 10000, device='cuda')
    start = time.time()
    y = x @ x
    torch.cuda.synchronize()  # Wait for GPU to finish
    end = time.time()
    print("Matrix multiplication time on GPU:", end - start)

