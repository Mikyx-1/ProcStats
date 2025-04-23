import os
import time
import psutil
import functools
import multiprocessing as mp


class MonitorCPUAndRAM:
    def __init__(self, interval=0.1):
        self.interval = interval

    def _monitor(self, pid):
        cpu_usages = []
        ram_usages = []

        try:
            proc = psutil.Process(pid)
            proc.cpu_percent(interval=None)  # Prime the CPU reading

            while proc.is_running():
                cpu_usages.append(proc.cpu_percent(interval=None))
                ram_usages.append(proc.memory_info().rss)
                time.sleep(self.interval)

        except psutil.NoSuchProcess:
            pass

        return {
            "cpu_max": max(cpu_usages, default=0),
            "cpu_avg": sum(cpu_usages) / len(cpu_usages) if cpu_usages else 0,
            "ram_max": max(ram_usages, default=0) / 1024**2,
            "ram_avg": sum(ram_usages) / len(ram_usages) / 1024**2 if ram_usages else 0,
        }


if __name__ == "__main__":
    cpu_and_ram_monitor = MonitorCPUAndRAM()
    res = cpu_and_ram_monitor._monitor(288287)
    print(res)