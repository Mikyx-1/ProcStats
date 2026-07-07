# Manual reference script, not runnable as-is: `combined_procstats` and
# `test_burn_cpu` are placeholders for whichever GPU-monitoring entry point
# and burn helper you wire up. Not collected by pytest.
import time

def example_function():
    # Simulate some CPU and GPU work
    print("Starting example function...")
    for _ in range(5):
        # Simulate computation
        _ = [x * x for x in range(1000000)]
        time.sleep(0.5)  # Simulate some delay
    print("Example function completed")

def heavy_gpu_task():
    import os

    import torch

    print("PID:", os.getpid())
    try:
        a = torch.randn(5000, 5000, device="cuda:1")
        for _ in range(5000):
            b = torch.matmul(a, a.T)
    except RuntimeError as e:
        print(f"GPU task failed: {e}")

# Monitor the example function with CPU, RAM, and GPU monitoring
result = combined_procstats.combined_resource_monitor(
    target=heavy_gpu_task,
    timeout=100.0,  # 10 seconds timeout
    interval=0.1   # 0.1 second sampling interval
)
print(f"Result: {result}")