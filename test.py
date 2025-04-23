import os
import time
import threading
import numpy as np


def burn_cpu(cpu_percent=50, duration=10):
    """
    Burn specific percentage of CPU for a duration.
    TODO: Fix the 'cannot exceed 100%' limitation
    """
    print(f"Spinning CPU at ~{cpu_percent}% for {duration}s")
    interval = 0.1  # 100ms
    busy_time = interval * (cpu_percent / 100.0)
    idle_time = interval - busy_time
    start_time = time.time()

    while time.time() - start_time < duration:
        busy_start = time.time()
        while time.time() - busy_start < busy_time:
            pass  # Burn CPU
        time.sleep(idle_time)


def allocate_ram(mb=500, duration=10):
    """
    Allocate RAM for a duration. Uses numpy to ensure actual RAM use.
    """
    print(f"Allocating ~{mb}MB RAM for {duration}s")
    arr = np.ones((mb * 1024 * 1024) // 8, dtype=np.float64)  # Each float64 = 8 bytes
    time.sleep(duration)
    return arr  # Just to avoid garbage collection


def controlled_cpu_ram_task(cpu_percent=50, ram_mb=500, duration=10):
    print("Task PID:", os.getpid())

    # Launch CPU burner in thread (to allow RAM allocation in main thread)
    cpu_thread = threading.Thread(target=burn_cpu, args=(cpu_percent, duration))
    cpu_thread.start()

    # Allocate RAM
    ram_data = allocate_ram(ram_mb, duration)

    cpu_thread.join()


controlled_cpu_ram_task(cpu_percent=95, ram_mb=300, duration=1000)
