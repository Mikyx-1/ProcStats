import os
import time

import numpy as np
from test_burn_cpu import burn_cpu_accurate

import procstats


def resource_heavy_task():
    """A test function that consumes CPU and RAM."""
    print(f"Child process PID: {os.getpid()}")

    # Allocate a large array to consume RAM
    large_array = np.random.rand(10000, 10000)  # ~800MB double array

    # Perform CPU-intensive computation
    for _ in range(50):
        result = np.dot(large_array, large_array.T)
        time.sleep(0.05)  # Simulate some processing delay

    return result


def main():
    print("Starting resource monitoring test...")

    # Test with default parameters (monitor CPU and RAM, 10s timeout)
    result = procstats.full_resource_monitor(
        target=burn_cpu_accurate, timeout=10.0, monitor="both"
    )

    print("\nMonitoring Results:")
    print(f"CPU Max Usage (%): {result['cpu_max']:.2f}")
    print(f"CPU Avg Usage (%): {result['cpu_avg']:.2f}")
    print(f"RAM Max Usage (MB): {result['ram_max']:.2f}")
    print(f"RAM Avg Usage (MB): {result['ram_avg']:.2f}")


if __name__ == "__main__":
    import os

    main()
