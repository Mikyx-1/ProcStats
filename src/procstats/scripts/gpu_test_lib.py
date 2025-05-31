import os

import procstats_pid
import torch

# Note: Make sure to install cloudpickle first: pip install cloudpickle


def heavy_gpu_task():
    """Example GPU-intensive task"""
    import os

    import torch

    print(f"Child PID: {os.getpid()}")
    try:
        # Create large tensors on GPU
        a = torch.randn(5000, 5000, device="cuda:1")
        result = None

        # Perform matrix multiplications
        for i in range(100):  # Reduced from 1000 for demo
            b = torch.matmul(a, a.T)
            if i == 99:  # Return result from last iteration
                result = b.sum().item()

        return {"status": "success", "final_sum": result, "pid": os.getpid()}

    except RuntimeError as e:
        print(f"GPU task failed: {e}")
        return {"status": "error", "error": str(e), "pid": os.getpid()}


def light_cpu_task():
    """Example CPU task for comparison - completely clean process"""
    import os
    import time

    print(f"CPU Task PID: {os.getpid()}")

    # This is a completely fresh process - no GPU context at all
    time.sleep(1)  # Simulate some work
    return {"status": "cpu_work_done", "value": 42, "pid": os.getpid()}


def check_gpu_task():
    """Task that checks GPU availability but doesn't use it"""
    import os

    import torch

    print(f"GPU Check PID: {os.getpid()}")

    available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if available else 0

    return {
        "status": "gpu_check_done",
        "cuda_available": available,
        "device_count": device_count,
        "pid": os.getpid(),
    }


print(f"Parent PID: {os.getpid()}")
print("=" * 60)

# Example 1: Monitor GPU-intensive function
print("=== Running GPU-intensive task ===")
result = procstats_pid.run_and_monitor_gpu_on_function(
    target=heavy_gpu_task, gpu_index=1, interval=0.01  # Use GPU 1 as in your example
)

print("GPU Monitoring Result:")
for key, value in result.items():
    if key == "samples":
        print(f"  {key}: {len(value)} samples, max={max(value) if value else 0}")
    else:
        print(f"  {key}: {value}")

print("\n" + "=" * 60 + "\n")

# Example 2: Monitor CPU task (should show ZERO GPU usage now!)
print("=== Running CPU-only task (separate process) ===")
result_cpu = procstats_pid.run_and_monitor_gpu_on_function(
    target=light_cpu_task, gpu_index=1, interval=0.01
)

print("CPU Task Monitoring Result:")
for key, value in result_cpu.items():
    if key == "samples":
        print(f"  {key}: {len(value)} samples")
    else:
        print(f"  {key}: {value}")

print("\n" + "=" * 60 + "\n")

# Example 3: Task that checks GPU but doesn't use it
print("=== Running GPU-check task (no actual GPU usage) ===")
result_check = procstats_pid.run_and_monitor_gpu_on_function(
    target=check_gpu_task, gpu_index=1, interval=0.01
)

print("GPU Check Task Monitoring Result:")
for key, value in result_check.items():
    if key == "samples":
        print(f"  {key}: {len(value)} samples")
    else:
        print(f"  {key}: {value}")

print("\n" + "=" * 60 + "\n")


# Example 4: Function with error handling
def failing_gpu_task():
    """Function that will fail"""
    import os

    import torch

    print(f"Failing Task PID: {os.getpid()}")
    # This will fail if CUDA is not available or wrong device
    a = torch.randn(100, 100, device="cuda:10")  # Non-existent GPU
    return a.sum()


print("=== Running failing task ===")
result_fail = procstats_pid.run_and_monitor_gpu_on_function(
    target=failing_gpu_task, gpu_index=0, interval=0.01
)

print("Failed Task Monitoring Result:")
for key, value in result_fail.items():
    print(f"  {key}: {value}")

print("\n" + "=" * 60 + "\n")

# Example 5: Simple lambda (clean process)
print("=== Running lambda function (separate process) ===")
result_lambda = procstats_pid.run_and_monitor_gpu_on_function(
    target=lambda: {
        "message": "Hello from separate process!",
        "pid": __import__("os").getpid(),
    },
    gpu_index=0,
    interval=0.01,
)

print("Lambda Result:")
print(f"  Function result: {result_lambda['function_result']}")
print(f"  GPU utilization: {result_lambda['gpu_util_mean']}")
print(f"  Monitored PID: {result_lambda['monitored_pid']}")

print("\n" + "=" * 60)
print("NOTE: Each task runs in a completely separate process!")
print("CPU-only tasks should now show ZERO GPU utilization.")
