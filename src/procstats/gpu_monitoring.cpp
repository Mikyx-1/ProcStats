#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <nvml.h>
#include <iostream>
#include <chrono>
#include <csignal>
#include <thread>
#include <vector>
#include <algorithm>
#include <numeric>
#include <unistd.h>
#include <sys/wait.h>

namespace py = pybind11;

struct GpuStats {
    double sm_util_avg = 0.0;
    double sm_util_max = 0.0;
    double vram_avg = 0.0;
    double vram_max = 0.0;
};

// Sample GPU SM utilization by PID
int sample_gpu_utilization(nvmlDevice_t handle, unsigned int pid) {
    unsigned int sampleCount = 0;
    // First call to get the required size
    nvmlReturn_t result = nvmlDeviceGetProcessUtilization(handle, nullptr, &sampleCount, 1000);
    if (result != NVML_SUCCESS && result != NVML_ERROR_INSUFFICIENT_SIZE) {
        std::cerr << "nvmlDeviceGetProcessUtilization (size query) failed: " << nvmlErrorString(result) << std::endl;
        return 0;
    }

    std::vector<nvmlProcessUtilizationSample_t> samples(sampleCount);
    result = nvmlDeviceGetProcessUtilization(handle, samples.data(), &sampleCount, 1000);
    if (result != NVML_SUCCESS) {
        std::cerr << "nvmlDeviceGetProcessUtilization failed: " << nvmlErrorString(result) << std::endl;
        // Fallback to overall GPU utilization
        nvmlUtilization_t utilization;
        result = nvmlDeviceGetUtilizationRates(handle, &utilization);
        if (result != NVML_SUCCESS) {
            std::cerr << "nvmlDeviceGetUtilizationRates failed: " << nvmlErrorString(result) << std::endl;
            return 0;
        }
        std::cerr << "Falling back to overall GPU utilization: " << utilization.gpu << "%" << std::endl;
        return utilization.gpu;
    }

    std::cerr << "Sample count: " << sampleCount << std::endl;
    for (unsigned int i = 0; i < sampleCount; ++i) {
        std::cerr << "Sample PID: " << samples[i].pid << ", SM Util: " << samples[i].smUtil << std::endl;
        if (samples[i].pid == pid) {
            return samples[i].smUtil;
        }
    }
    std::cerr << "PID " << pid << " not found in utilization samples" << std::endl;
    return 0;
}

// Sample GPU VRAM usage by PID
double sample_vram_usage(nvmlDevice_t handle, unsigned int pid) {
    unsigned int infoCount = 0;
    // First call to get the required size
    nvmlReturn_t result = nvmlDeviceGetComputeRunningProcesses(handle, &infoCount, nullptr);
    if (result != NVML_SUCCESS && result != NVML_ERROR_INSUFFICIENT_SIZE) {
        std::cerr << "nvmlDeviceGetComputeRunningProcesses (size query) failed: " << nvmlErrorString(result) << std::endl;
        return 0.0;
    }

    std::vector<nvmlProcessInfo_t> infos(infoCount);
    result = nvmlDeviceGetComputeRunningProcesses(handle, &infoCount, infos.data());
    if (result != NVML_SUCCESS) {
        std::cerr << "nvmlDeviceGetComputeRunningProcesses failed: " << nvmlErrorString(result) << std::endl;
        return 0.0;
    }

    std::cerr << "VRAM sample count: " << infoCount << std::endl;
    for (unsigned int i = 0; i < infoCount; ++i) {
        std::cerr << "VRAM Sample PID: " << infos[i].pid << ", Used Memory: " << infos[i].usedGpuMemory / 1024.0 / 1024.0 << " MB" << std::endl;
        if (infos[i].pid == pid) {
            return infos[i].usedGpuMemory / 1024.0 / 1024.0; // MB
        }
    }
    std::cerr << "PID " << pid << " not found in VRAM samples" << std::endl;
    return 0.0;
}

GpuStats monitor_gpu(unsigned int gpu_index, pid_t pid, double interval_sec, int max_samples = 100) {
    GpuStats stats;
    std::vector<int> sm_utils;
    std::vector<double> vram_usages;

    std::cerr << "Monitoring GPU " << gpu_index << " for PID " << pid << std::endl;
    nvmlReturn_t result = nvmlInit_v2();
    if (result != NVML_SUCCESS) {
        std::cerr << "nvmlInit_v2 failed: " << nvmlErrorString(result) << std::endl;
        return stats;
    }

    nvmlDevice_t device;
    if (nvmlDeviceGetHandleByIndex(gpu_index, &device) != NVML_SUCCESS) {
        std::cerr << "nvmlDeviceGetHandleByIndex failed" << std::endl;
        nvmlShutdown();
        return stats;
    }

    int sample_count = 0;
    while (sample_count < max_samples) {
        int status;
        pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid || kill(pid, 0) != 0) {
            std::cerr << "Process " << pid << " terminated or not found" << std::endl;
            break;
        }

        int sm_util = sample_gpu_utilization(device, pid);
        double vram = sample_vram_usage(device, pid);

        sm_utils.push_back(sm_util);
        vram_usages.push_back(vram);

        std::this_thread::sleep_for(std::chrono::duration<double>(interval_sec));
        sample_count++;
    }

    nvmlShutdown();

    if (!sm_utils.empty()) {
        stats.sm_util_avg = std::accumulate(sm_utils.begin(), sm_utils.end(), 0.0) / sm_utils.size();
        stats.sm_util_max = *std::max_element(sm_utils.begin(), sm_utils.end());
    }
    if (!vram_usages.empty()) {
        stats.vram_avg = std::accumulate(vram_usages.begin(), vram_usages.end(), 0.0) / vram_usages.size();
        stats.vram_max = *std::max_element(vram_usages.begin(), vram_usages.end());
    }

    return stats;
}

py::dict gpu_monitor(py::function py_func, int gpu_index = 0, double interval = 0.1, int max_samples = 100) {
    pid_t pid = fork();
    if (pid == 0) {
        py_func();
        _exit(0);
    }

    std::cerr << "Parent process monitoring PID: " << pid << std::endl;
    GpuStats stats = monitor_gpu(gpu_index, pid, interval, max_samples);
    waitpid(pid, nullptr, 0);

    py::dict result;
    result["gpu_util_mean"] = stats.sm_util_avg;
    result["gpu_util_max"] = stats.sm_util_max;
    result["vram_usage_mean"] = stats.vram_avg;
    result["vram_usage_max"] = stats.vram_max;
    return result;
}

PYBIND11_MODULE(procstats_gpu, m) {
    m.def("gpu_monitor", &gpu_monitor,
          py::arg("target"),
          py::arg("gpu_index") = 0,
          py::arg("interval") = 0.1,
          py::arg("max_samples") = 100,
          "Monitors GPU usage (SM utilization and VRAM) for a target Python function.");
}