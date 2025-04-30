import multiprocessing as mp
import os
import time
from typing import Callable

from cpu_ram_monitoring import monitor_cpu_and_ram_by_pid
from gpu_monitoring import monitor_gpu_utilization_by_pid


def full_resource_monitor(
    target: Callable, args=(), kwargs=None, interval=0.01, gpu_index=0
):

    if kwargs is None:
        kwargs = {}
    mp.set_start_method("spawn", force=True)

    cpu_and_ram_result_container = mp.Manager().list()
    gpu_result_container = mp.Manager().list()

    # Launch processes
    process = mp.Process(target=target, args=args, kwargs=kwargs)
    process.start()

    cpu_and_ram_monitor_process = mp.Process(
        target=monitor_cpu_and_ram_by_pid,
        args=(process.pid, interval, cpu_and_ram_result_container),
    )
    gpu_monitor_process = mp.Process(
        target=monitor_gpu_utilization_by_pid,
        args=(gpu_index, process.pid, interval, gpu_result_container),
    )

    # Start processes
    cpu_and_ram_monitor_process.start()
    gpu_monitor_process.start()

    # Join processes
    process.join()
    cpu_and_ram_monitor_process.join()
    gpu_monitor_process.join()

    return cpu_and_ram_result_container[0], gpu_result_container[0]


def heavy_cpu_gpu_task():
    import os

    import torch

    time.sleep(2)
    print("Inside PID:", os.getpid())
    a = torch.randn(5000, 5000, device="cuda:1")
    for _ in range(10):
        b = torch.matmul(a, a.T)


if __name__ == "__main__":
    res = full_resource_monitor(heavy_cpu_gpu_task, gpu_index=1)
    print(res)
