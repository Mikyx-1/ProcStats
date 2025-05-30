#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nvml.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <thread>
#include <chrono>
#include <iostream>

namespace py = pybind11;

struct GpuUtilStats {
    double gpu_util_mean = 0.0;
    unsigned int gpu_util_max = 0;
    bool process_found = false;
    bool process_has_memory = false;
    unsigned long long memory_used = 0;
    std::string error_msg = "";
    std::string method_used = "";
    std::vector<double> utilization_samples;
};

// Method 1: Enhanced NVML with multiple sampling strategies
GpuUtilStats method1_enhanced_nvml(unsigned int pid, unsigned int gpu_index, unsigned int num_samples, double interval_seconds) {
    GpuUtilStats result;
    result.method_used = "Enhanced NVML";
    
    nvmlReturn_t nvml_result = nvmlInit_v2();
    if (nvml_result != NVML_SUCCESS) {
        result.error_msg = "NVML init failed: " + std::string(nvmlErrorString(nvml_result));
        return result;
    }

    nvmlDevice_t device;
    nvml_result = nvmlDeviceGetHandleByIndex(gpu_index, &device);
    if (nvml_result != NVML_SUCCESS) {
        result.error_msg = "Failed to get device: " + std::string(nvmlErrorString(nvml_result));
        nvmlShutdown();
        return result;
    }

    // Check memory allocation
    unsigned int processCount = 128;
    nvmlProcessInfo_t processes[128];
    nvml_result = nvmlDeviceGetComputeRunningProcesses(device, &processCount, processes);
    
    if (nvml_result == NVML_SUCCESS) {
        for (unsigned int i = 0; i < processCount; ++i) {
            if (processes[i].pid == pid) {
                result.process_has_memory = true;
                result.memory_used = processes[i].usedGpuMemory;
                break;
            }
        }
    }

    std::vector<unsigned int> sm_utils;
    std::vector<unsigned long long> time_windows = {500, 1000, 2000, 5000, 10000}; // microseconds

    for (unsigned int sample = 0; sample < num_samples; ++sample) {
        bool found_in_this_sample = false;
        
        for (auto time_window : time_windows) {
            unsigned int sampleCount = 128;
            nvmlProcessUtilizationSample_t samples[128];
            
            nvml_result = nvmlDeviceGetProcessUtilization(device, samples, &sampleCount, time_window);
            if (nvml_result == NVML_SUCCESS && sampleCount > 0) {
                for (unsigned int j = 0; j < sampleCount; ++j) {
                    if (samples[j].pid == pid && samples[j].smUtil > 0) {
                        sm_utils.push_back(samples[j].smUtil);
                        result.process_found = true;
                        found_in_this_sample = true;
                        break;
                    }
                }
                if (found_in_this_sample) break;
            }
        }

        if (sample < num_samples - 1) {
            std::this_thread::sleep_for(std::chrono::duration<double>(interval_seconds));
        }
    }

    if (!sm_utils.empty()) {
        result.gpu_util_max = *std::max_element(sm_utils.begin(), sm_utils.end());
        result.gpu_util_mean = std::accumulate(sm_utils.begin(), sm_utils.end(), 0.0) / sm_utils.size();
        result.utilization_samples.assign(sm_utils.begin(), sm_utils.end());
    }

    nvmlShutdown();
    return result;
}

// Pybind wrapper
py::dict monitor_gpu_util_by_pid_py(unsigned int pid, unsigned int gpu_index = 0, unsigned int num_samples = 10, double interval_seconds = 0.1) {
    GpuUtilStats stats = method1_enhanced_nvml(pid, gpu_index, num_samples, interval_seconds);
    py::dict result;
    result["gpu_util_mean"] = stats.gpu_util_mean;
    result["gpu_util_max"] = stats.gpu_util_max;
    result["process_found"] = stats.process_found;
    result["process_has_memory"] = stats.process_has_memory;
    result["memory_used_mb"] = stats.memory_used / (1024 * 1024);
    result["error_msg"] = stats.error_msg;
    result["method_used"] = stats.method_used;
    result["num_samples"] = static_cast<int>(stats.utilization_samples.size());
    
    py::list samples_list;
    for (auto sample : stats.utilization_samples) {
        samples_list.append(sample);
    }
    result["samples"] = samples_list;
    
    return result;
}

py::list list_gpu_processes_py(unsigned int gpu_index = 0) {
    py::list result;
    
    if (nvmlInit_v2() != NVML_SUCCESS) {
        return result;
    }

    nvmlDevice_t device;
    if (nvmlDeviceGetHandleByIndex(gpu_index, &device) != NVML_SUCCESS) {
        nvmlShutdown();
        return result;
    }

    unsigned int processCount = 128;
    nvmlProcessInfo_t processes[128];
    if (nvmlDeviceGetComputeRunningProcesses(device, &processCount, processes) == NVML_SUCCESS) {
        for (unsigned int i = 0; i < processCount; ++i) {
            py::dict proc_info;
            proc_info["pid"] = processes[i].pid;
            proc_info["memory_mb"] = processes[i].usedGpuMemory / (1024 * 1024);
            proc_info["type"] = "memory_allocated";
            result.append(proc_info);
        }
    }

    nvmlShutdown();
    return result;
}

PYBIND11_MODULE(procstats_pid, m) {
    m.def("monitor_gpu_util_by_pid", &monitor_gpu_util_by_pid_py,
          py::arg("pid"), py::arg("gpu_index") = 0,
          py::arg("num_samples") = 10, py::arg("interval_seconds") = 0.1,
          "Monitor GPU SM Utilization by PID using Enhanced NVML");

    m.def("list_gpu_processes", &list_gpu_processes_py,
          py::arg("gpu_index") = 0,
          "List all processes using GPU");
}
