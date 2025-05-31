import math
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import numpy as np


def pure_cpu_burn(duration):
    """
    Pure CPU burning without any sleep - maxes out one core.
    """
    start_time = time.time()
    counter = 0

    while time.time() - start_time < duration:
        # Intensive CPU work - no sleep at all
        counter += 1
        # Some actual computation to prevent optimization
        x = math.sqrt(counter % 10000 + 1)
        y = math.sin(x) * math.cos(x)
        z = y**0.5 if y > 0 else 0


def controlled_cpu_burn_process(target_percent, duration, process_id):
    """
    CPU burn for a specific percentage in a separate process.
    Uses busy-wait with precise timing control.
    """
    if target_percent <= 0:
        return

    print(f"Process {process_id}: Starting {target_percent}% CPU burn for {duration}s")

    # Use very small time slices for accuracy
    slice_duration = 0.01  # 10ms slices
    busy_ratio = min(target_percent / 100.0, 1.0)
    busy_time = slice_duration * busy_ratio
    idle_time = slice_duration - busy_time

    start_time = time.time()

    while time.time() - start_time < duration:
        # Busy period - pure CPU work
        busy_start = time.time()
        counter = 0
        while time.time() - busy_start < busy_time:
            counter += 1
            # Prevent compiler optimization
            x = math.sqrt(counter % 1000 + 1)
            y = x * 1.414213562373095

        # Idle period - but only if needed
        if idle_time > 0:
            time.sleep(idle_time)


def burn_cpu_accurate(cpu_percent=50, duration=10):
    """
    Accurate CPU burning that can exceed 100% using multiple processes.

    Args:
        cpu_percent: Target CPU percentage (can exceed 100%)
        duration: Duration in seconds
    """
    num_cores = multiprocessing.cpu_count()
    max_cpu = num_cores * 100

    print(f"System: {num_cores} CPU cores (theoretical max {max_cpu}%)")
    print(f"Target: {cpu_percent}% CPU for {duration}s")

    if cpu_percent <= 0:
        return

    if cpu_percent > max_cpu:
        print(f"Warning: {cpu_percent}% exceeds system maximum {max_cpu}%")

    # Strategy: Use processes instead of threads for true parallelism
    if cpu_percent <= 100:
        # Single process with controlled load
        processes = [(cpu_percent, 0)]
    else:
        # Multiple processes
        full_cores = int(cpu_percent // 100)
        remainder = cpu_percent % 100

        processes = []

        # Full-load processes
        for i in range(min(full_cores, num_cores)):
            processes.append((100, i))

        # Partial load process for remainder
        if remainder > 0 and len(processes) < num_cores:
            processes.append((remainder, len(processes)))

    print(f"Strategy: {len(processes)} processes")
    for percent, pid in processes:
        print(f"  Process {pid}: {percent}%")

    # Use ProcessPoolExecutor for better control
    with ProcessPoolExecutor(max_workers=len(processes)) as executor:
        futures = []
        for percent, proc_id in processes:
            future = executor.submit(
                controlled_cpu_burn_process, percent, duration, proc_id
            )
            futures.append(future)

        # Wait for all processes to complete
        for future in futures:
            try:
                future.result(timeout=duration + 5)
            except Exception as e:
                print(f"Process error: {e}")


def burn_cpu_maximum(duration=10):
    """
    Maximum CPU burn - 100% on all available cores using pure busy loops.
    """
    num_cores = multiprocessing.cpu_count()
    print(f"Maximum CPU burn: {num_cores} cores at 100% for {duration}s")

    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for i in range(num_cores):
            future = executor.submit(pure_cpu_burn, duration)
            futures.append(future)

        for future in futures:
            future.result()


def allocate_ram(mb=500, duration=10):
    """
    Allocate and actively use RAM to ensure it's actually consumed.
    """
    print(f"Allocating {mb}MB RAM for {duration}s")

    try:
        # Calculate elements needed
        elements = (mb * 1024 * 1024) // 8  # 8 bytes per float64

        # Allocate array
        arr = np.random.random(elements).astype(np.float64)

        # Actively use the memory to prevent optimization
        start_time = time.time()
        while time.time() - start_time < duration:
            # Touch random parts of the array
            indices = np.random.randint(0, len(arr), size=min(1000, len(arr)))
            arr[indices] *= 1.0001
            time.sleep(0.1)  # Small sleep to avoid excessive CPU from memory ops

        return arr

    except MemoryError:
        print(f"Failed to allocate {mb}MB - insufficient memory")
        return None


def controlled_cpu_ram_task(cpu_percent=100, ram_mb=1000, duration=10):
    """
    Accurate controlled task with proper CPU and RAM usage.
    """
    print("=" * 60)
    print(f"Task PID: {os.getpid()}")
    print(f"Target: {cpu_percent}% CPU, {ram_mb}MB RAM for {duration}s")
    print("=" * 60)

    # Start RAM allocation in a separate thread
    ram_thread = threading.Thread(target=lambda: allocate_ram(ram_mb, duration))
    ram_thread.start()

    # Run CPU burn (this will spawn processes)
    burn_cpu_accurate(cpu_percent, duration)

    # Wait for RAM thread
    ram_thread.join()

    print("Task completed!")


def stress_test_validation():
    """
    Run validation tests to check accuracy.
    """
    print("Starting CPU stress test validation...")
    print("Monitor with 'htop', 'top', or similar tools")

    test_cases = [
        (25, 10),  # 25% single core
        (100, 10),  # 100% single core
        (200, 10),  # 200% (2 cores)
        (400, 10),  # 400% (4 cores)
    ]

    for cpu_pct, dur in test_cases:
        print(f"\n{'='*50}")
        print(f"TEST: {cpu_pct}% CPU for {dur} seconds")
        print(f"{'='*50}")

        input("Press Enter to start this test (monitor CPU usage)...")
        burn_cpu_accurate(cpu_pct, dur)

        print("Test completed. Check if CPU usage matched target.")
        time.sleep(2)


# Alternative implementation using threading for finer control
def threaded_cpu_burn(target_percent, duration):
    """
    Thread-based CPU burn with more precise control.
    """

    def cpu_worker(load_percent, work_duration):
        if load_percent >= 100:
            # Pure burn for 100%
            pure_cpu_burn(work_duration)
        else:
            # Controlled burn
            controlled_cpu_burn_process(load_percent, work_duration, 0)

    num_cores = multiprocessing.cpu_count()

    if target_percent <= 100:
        # Single thread
        thread = threading.Thread(target=cpu_worker, args=(target_percent, duration))
        thread.start()
        thread.join()
    else:
        # Multiple threads
        threads_needed = min(math.ceil(target_percent / 100), num_cores)
        per_thread_load = min(target_percent / threads_needed, 100)

        threads = []
        for i in range(threads_needed):
            thread = threading.Thread(
                target=cpu_worker, args=(per_thread_load, duration)
            )
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()


if __name__ == "__main__":
    # Quick test
    print("Quick CPU burn test - monitor with htop/nvitop")
    burn_cpu_accurate(cpu_percent=300, duration=20)
