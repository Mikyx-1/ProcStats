import multiprocessing as mp
import os
import time
from typing import Callable

from cpu_ram_monitoring import MonitorCPUAndRAM
from gpu_monitoring import MonitorGPU


class FullSystemMonitor:
    def __init__(self, interval=0.1, gpu_index=0, use_gpu=True):
        self.interval = interval
        self.gpu_index = gpu_index
        self.use_gpu = use_gpu

        self.monitor_gpu = MonitorGPU(gpu_index=self.gpu_index, interval=self.interval)
        self.monitor_cpu_and_ram = MonitorCPUAndRAM(interval=self.interval)

    def monitor(self, target: Callable, args=(), kwargs=None):

        if kwargs is None:
            kwargs = {}
        mp.set_start_method("spawn", force=True)

        cpu_and_ram_result_container = mp.Manager().list()
        gpu_result_container = mp.Manager().list()

        # Launch the target process
        process = mp.Process(target=target, args=args, kwargs=kwargs)
        process.start()

        # Launch the cpu & ram monitor process
        cpu_and_ram_monitor_process = mp.Process(
            target=self.monitor_cpu_and_ram.monitor_until_static,
            args=(process.pid, self.interval, cpu_and_ram_result_container),
        )
        gpu_monitor_process = mp.Process(
            target=self.monitor_gpu.monitor_until_static,
            args=(self.gpu_index, process.pid, self.interval, gpu_result_container),
        )

        cpu_and_ram_monitor_process.start()
        gpu_monitor_process.start()

        # Wait for all to join
        process.join()
        cpu_and_ram_monitor_process.join()
        gpu_monitor_process.join()

        return cpu_and_ram_result_container[0], gpu_result_container[0]

def heavy_cpu_gpu_task():
    import os

    import torch

    time.sleep(2)
    print("Inside PID:", os.getpid())
    a = torch.randn(5000, 50000, device="cuda:1")
    for _ in range(10):
        b = torch.matmul(a, a.T)


if __name__ == "__main__":
    full_system_monitor = FullSystemMonitor(interval=0.01, gpu_index=1, use_gpu=True)
    res = full_system_monitor.monitor(target=heavy_cpu_gpu_task)
    print(res)