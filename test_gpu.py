
import torch, time

def benchmark(device, size=4096, repeats=100):
    print(f"\nRunning on {device}...")
    x = torch.randn(size, size, device=device)

    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(repeats):
        y = x @ x
    if device == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - t0
    print(f"{device} time for {repeats} matmuls of {size}x{size}: {elapsed:.4f} seconds")
    return elapsed

if __name__ == "__main__":
    # Always run CPU
    cpu_time = benchmark("cpu")

    # Run GPU if available
    if torch.cuda.is_available():
        gpu_time = benchmark("cuda")

        speedup = cpu_time / gpu_time
        print(f"\nSpeedup (CPU/GPU): {speedup:.1f}x")
    else:
        print("\nNo GPU detected.")
