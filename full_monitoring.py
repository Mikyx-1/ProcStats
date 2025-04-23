import multiprocessing as mp
import os
import time

from cpu_ram_monitoring import MonitorCPUAndRAM
from gpu_monitoring import MonitorGPU


class FullSystemMonitor:
    def __init__(self, interval=0.1, gpu_index=0, use_gpu=True):
        self.interval = interval
        self.gpu_index = gpu_index
        self.use_gpu = use_gpu

    def _monitor_all(self, pid, stats_dict):
        cpu_ram_monitor = MonitorCPUAndRAM(interval=self.interval)
        cpu_ram_stats = cpu_ram_monitor._monitor(pid=pid)

        if self.use_gpu:
            gpu_monitor = MonitorGPU(interval=self.interval)
            gpu_stats = gpu_monitor._monitor(pid=pid, gpu_index=self.gpu_index)
        else:
            gpu_stats = {"gpu_utilisations": [], "gpu_vram_usages": []}

        stats_dict["cpu_usages"] = cpu_ram_stats["cpu_usages"]
        stats_dict["ram_usages"] = cpu_ram_stats["ram_usages"]
        stats_dict["gpu_utilisations"] = gpu_stats["gpu_utilisations"]
        stats_dict["gpu_vram_usages"] = gpu_stats["gpu_vram_usages"]

    def monitor_function(self, func, *args, **kwargs):
        manager = mp.Manager()
        stats = manager.dict()

        def target(return_dict, *a, **kw):
            result = func(*a, **kw)
            return_dict["result"] = result

        return_dict = manager.dict()
        proc = mp.Process(target=target, args=(return_dict, *args), kwargs=kwargs)
        proc.start()

        monitor_proc = mp.Process(target=self._monitor_all, args=(proc.pid, stats))
        monitor_proc.start()

        proc.join()
        monitor_proc.terminate()  # Ends monitor after function exits

        return {
            "result": return_dict.get("result"),
            "cpu_usages": list(stats.get("cpu_usages", [])),
            "ram_usages": list(stats.get("ram_usages", [])),
            "gpu_utilisations": list(stats.get("gpu_utilisations", [])),
            "gpu_vram_usages": list(stats.get("gpu_vram_usages", [])),
        }


# Example usage
if __name__ == "__main__":

    def heavy_cpu_gpu_task():
        import torch

        print("Inside PID:", os.getpid())
        a = torch.randn(5000, 5000, device="cuda:0")
        for _ in range(5):
            b = torch.matmul(a, a.T)
            time.sleep(2)

    monitor = FullSystemMonitor(interval=0.1, gpu_index=0, use_gpu=True)
    output = monitor.monitor_function(heavy_cpu_gpu_task)

    print("--- Monitoring Summary ---")
    print("CPU usage samples:", output["cpu_usages"])
    print("RAM usage samples (MB):", [v / 1024**2 for v in output["ram_usages"]])
    print("GPU utilisation samples:", output["gpu_utilisations"])
    print("GPU VRAM usage samples (MB):", output["gpu_vram_usages"])
