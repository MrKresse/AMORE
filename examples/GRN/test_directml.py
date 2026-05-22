import torch
print("torch version:", torch.__version__)

import torch_directml
dml = torch_directml.device()
print("DirectML device:", dml)

# Basic tensor op
a = torch.randn(1024, 1024).to(dml)
b = torch.randn(1024, 1024).to(dml)
c = a @ b
print("Matrix multiply OK, result shape:", c.shape)

# Verify torch still works on CPU
x = torch.randn(10)
print("CPU still works:", x.sum().item())

# Benchmark: large matmul CPU vs DirectML
import time

sizes = [(2048, 2000), (2048, 1024)]
for m, n in sizes:
    a_cpu = torch.randn(m, n)
    b_cpu = torch.randn(n, m)
    t0 = time.perf_counter()
    for _ in range(20):
        _ = a_cpu @ b_cpu
    t_cpu = (time.perf_counter() - t0) / 20 * 1000

    a_dml = a_cpu.to(dml)
    b_dml = b_cpu.to(dml)
    # Warm-up
    for _ in range(3): _ = a_dml @ b_dml
    torch.dml.synchronize() if hasattr(torch, 'dml') else None
    t0 = time.perf_counter()
    for _ in range(20):
        _ = a_dml @ b_dml
    t_dml = (time.perf_counter() - t0) / 20 * 1000

    print(f"  ({m}x{n})@({n}x{m}):  CPU={t_cpu:.1f}ms  DirectML={t_dml:.1f}ms  "
          f"speedup={t_cpu/t_dml:.1f}x")

print("\nAll tests passed.")
