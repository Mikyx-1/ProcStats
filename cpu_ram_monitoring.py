import functools
import multiprocessing as mp
import os
import time

import psutil


class MonitorCPUAndRAM:
    def __init__(self, interval=0.1):
        self.interval = interval

    @staticmethod
    def monitor_until_static(pid, interval, result_container):
        cpu_usages = []
        ram_usages = []

        try:
            proc = psutil.Process(pid)
            proc.cpu_percent(interval=interval)

            while psutil.pid_exists(pid) and psutil.Process(pid).is_running():
                cpu_percent = proc.cpu_percent(interval=None)
                ram_usage = proc.memory_info().rss
                cpu_usages.append(cpu_percent)
                ram_usages.append(ram_usage)
                time.sleep(interval)

        finally:
            result_container.append(
                {
                    "cpu_max": max(cpu_usages, default=0),
                    "cpu_avg": sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0,
                    "ram_max": max(ram_usages, default=0) / 1024**2,
                    "ram_avg": (
                        sum(ram_usages) / len(ram_usages) / 1024**2 if ram_usages else 0
                    ),
                }
            )

    def monitor(self, target, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}

        mp.set_start_method("spawn", force=True)
        manager = mp.Manager()
        result_container = manager.list()

        # Launch target process
        process = mp.Process(target=target, args=args, kwargs=kwargs)
        process.start()

        # Launch monitor process
        monitor_proc = mp.Process(
            target=MonitorCPUAndRAM.monitor_until_static,
            args=(process.pid, self.interval, result_container),
        )

        monitor_proc.start()

        # Wait for both
        process.join()
        monitor_proc.join()

        return (
            result_container[0]
            if result_container
            else {
                "cpu_max": 0,
                "cpu_avg": 0,
                "ram_max": 0,
                "ram_avg": 0,
            }
        )


def heavy_cpu_gpu_task():
    import os

    import torch

    print("Inside PID:", os.getpid())
    a = torch.randn(5000, 5000)
    for _ in range(10):
        b = torch.matmul(a, a.T)


if __name__ == "__main__":
    cpu_and_ram_monitor = MonitorCPUAndRAM(interval=0.01)
    resource_usg = cpu_and_ram_monitor.monitor(heavy_cpu_gpu_task)
    print(resource_usg)
