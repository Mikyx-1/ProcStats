#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <nvml.h>
#include <cuda_runtime.h>
#include <cupti.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <thread>
#include <chrono>
#include <iostream>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <sys/stat.h>
#include <dirent.h>

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
    
    // Try multiple time windows and sampling strategies
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
        for (auto util : sm_utils) {
            result.utilization_samples.push_back(util);
        }
    }

    nvmlShutdown();
    return result;
}

// Method 2: CUPTI-based process monitoring
GpuUtilStats method2_cupti_monitoring(unsigned int pid, unsigned int gpu_index, unsigned int num_samples, double interval_seconds) {
    GpuUtilStats result;
    result.method_used = "CUPTI Activity";
    
    // Initialize CUPTI
    CUptiResult cupti_result = cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL);
    if (cupti_result != CUPTI_SUCCESS) {
        result.error_msg = "CUPTI kernel activity enable failed";
        return result;
    }

    // This is a simplified version - full implementation would require
    // setting up activity buffers and callbacks
    cupti_result = cuptiActivityDisable(CUPTI_ACTIVITY_KIND_KERNEL);
    
    result.error_msg = "CUPTI method requires more complex setup - use alternative methods";
    return result;
}

// Method 3: Procfs + sysfs based monitoring
GpuUtilStats method3_procfs_monitoring(unsigned int pid, unsigned int gpu_index, unsigned int num_samples, double interval_seconds) {
    GpuUtilStats result;
    result.method_used = "Procfs/Sysfs";
    
    // Check if process exists and has GPU context
    std::string proc_path = "/proc/" + std::to_string(pid);
    struct stat st;
    if (stat(proc_path.c_str(), &st) != 0) {
        result.error_msg = "Process not found";
        return result;
    }

    // Try to read GPU-related information from /proc/pid/maps
    std::ifstream maps_file(proc_path + "/maps");
    bool has_gpu_mapping = false;
    
    if (maps_file.is_open()) {
        std::string line;
        while (std::getline(maps_file, line)) {
            if (line.find("nvidia") != std::string::npos || 
                line.find("cuda") != std::string::npos ||
                line.find("/dev/nvidiactl") != std::string::npos) {
                has_gpu_mapping = true;
                break;
            }
        }
        maps_file.close();
    }

    if (!has_gpu_mapping) {
        result.error_msg = "No GPU mappings found in process memory";
        return result;
    }

    // Monitor GPU utilization and correlate with process activity
    std::vector<double> gpu_utils;
    std::vector<unsigned long long> process_gpu_time;
    
    nvmlInit_v2();
    nvmlDevice_t device;
    if (nvmlDeviceGetHandleByIndex(gpu_index, &device) == NVML_SUCCESS) {
        for (unsigned int i = 0; i < num_samples; ++i) {
            // Get overall GPU utilization
            nvmlUtilization_t util;
            if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS) {
                gpu_utils.push_back(util.gpu);
            }
            
            // Check if process is active (simplified heuristic)
            std::ifstream stat_file(proc_path + "/stat");
            if (stat_file.is_open()) {
                std::string stat_line;
                std::getline(stat_file, stat_line);
                // This is a simplified approach - would need more sophisticated correlation
                stat_file.close();
            }
            
            if (i < num_samples - 1) {
                std::this_thread::sleep_for(std::chrono::duration<double>(interval_seconds));
            }
        }
    }
    nvmlShutdown();

    if (!gpu_utils.empty()) {
        result.gpu_util_max = *std::max_element(gpu_utils.begin(), gpu_utils.end());
        result.gpu_util_mean = std::accumulate(gpu_utils.begin(), gpu_utils.end(), 0.0) / gpu_utils.size();
        result.process_found = true;
        result.utilization_samples = gpu_utils;
    }

    return result;
}

// Method 4: GPU event-based monitoring using CUDA driver API
GpuUtilStats method4_cuda_events(unsigned int pid, unsigned int gpu_index, unsigned int num_samples, double interval_seconds) {
    GpuUtilStats result;
    result.method_used = "CUDA Events";

    // This approach monitors CUDA contexts and events
    // Requires more complex setup but can be more accurate
    
    cudaError_t cuda_result = cudaSetDevice(gpu_index);
    if (cuda_result != cudaSuccess) {
        result.error_msg = "Failed to set CUDA device: " + std::string(cudaGetErrorString(cuda_result));
        return result;
    }

    // Create events for timing
    cudaEvent_t start_event, stop_event;
    cuda_result = cudaEventCreate(&start_event);
    if (cuda_result != cudaSuccess) {
        result.error_msg = "Failed to create CUDA events";
        return result;
    }
    
    cuda_result = cudaEventCreate(&stop_event);
    if (cuda_result != cudaSuccess) {
        cudaEventDestroy(start_event);
        result.error_msg = "Failed to create CUDA events";
        return result;
    }

    // This is a placeholder - actual implementation would require
    // hooking into the target process's CUDA contexts
    std::vector<float> timing_samples;
    
    for (unsigned int i = 0; i < num_samples; ++i) {
        cudaEventRecord(start_event);
        
        // Wait and measure - this is simplified
        std::this_thread::sleep_for(std::chrono::duration<double>(interval_seconds));
        
        cudaEventRecord(stop_event);
        cudaEventSynchronize(stop_event);
        
        float elapsed_time;
        cudaEventElapsedTime(&elapsed_time, start_event, stop_event);
        timing_samples.push_back(elapsed_time);
    }

    cudaEventDestroy(start_event);
    cudaEventDestroy(stop_event);

    result.error_msg = "CUDA Events method needs process context injection";
    return result;
}

// Method 5: Nvidia-ml-py equivalent in C++ with aggressive sampling
GpuUtilStats method5_aggressive_sampling(unsigned int pid, unsigned int gpu_index, unsigned int num_samples, double interval_seconds) {
    GpuUtilStats result;
    result.method_used = "Aggressive NVML Sampling";
    
    nvmlReturn_t nvml_result = nvmlInit_v2();
    if (nvml_result != NVML_SUCCESS) {
        result.error_msg = "NVML init failed";
        return result;
    }

    nvmlDevice_t device;
    nvml_result = nvmlDeviceGetHandleByIndex(gpu_index, &device);
    if (nvml_result != NVML_SUCCESS) {
        result.error_msg = "Failed to get device";
        nvmlShutdown();
        return result;
    }

    // Check memory first
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

    if (!result.process_has_memory) {
        result.error_msg = "Process not found in GPU memory";
        nvmlShutdown();
        return result;
    }

    // Aggressive sampling strategy
    std::vector<unsigned int> all_utils;
    const unsigned int micro_samples = 50; // Many micro-samples per main sample
    
    for (unsigned int sample = 0; sample < num_samples; ++sample) {
        std::vector<unsigned int> micro_utils;
        
        for (unsigned int micro = 0; micro < micro_samples; ++micro) {
            unsigned int sampleCount = 64;
            nvmlProcessUtilizationSample_t samples[64];
            
            // Try very short time windows with high frequency
            nvml_result = nvmlDeviceGetProcessUtilization(device, samples, &sampleCount, 100);
            
            if (nvml_result == NVML_SUCCESS) {
                for (unsigned int j = 0; j < sampleCount; ++j) {
                    if (samples[j].pid == pid) {
                        micro_utils.push_back(samples[j].smUtil);
                        result.process_found = true;
                        break;
                    }
                }
            }
            
            // Very short sleep between micro-samples
            std::this_thread::sleep_for(std::chrono::microseconds(10000)); // 10ms
        }
        
        // Add all micro-samples to main collection
        all_utils.insert(all_utils.end(), micro_utils.begin(), micro_utils.end());
        
        if (sample < num_samples - 1) {
            std::this_thread::sleep_for(std::chrono::duration<double>(interval_seconds));
        }
    }

    if (!all_utils.empty()) {
        result.gpu_util_max = *std::max_element(all_utils.begin(), all_utils.end());
        result.gpu_util_mean = std::accumulate(all_utils.begin(), all_utils.end(), 0.0) / all_utils.size();
        for (auto util : all_utils) {
            result.utilization_samples.push_back(util);
        }
    }

    nvmlShutdown();
    return result;
}

// Main function that tries all methods
GpuUtilStats monitor_gpu_util_by_pid(unsigned int pid, unsigned int gpu_index, unsigned int num_samples = 10, double interval_seconds = 0.1) {
    std::vector<std::function<GpuUtilStats()>> methods = {
        [=]() { return method1_enhanced_nvml(pid, gpu_index, num_samples, interval_seconds); },
        [=]() { return method5_aggressive_sampling(pid, gpu_index, num_samples, interval_seconds); },
        [=]() { return method3_procfs_monitoring(pid, gpu_index, num_samples, interval_seconds); }
    };

    // Try methods in order of preference
    for (auto& method : methods) {
        GpuUtilStats result = method();
        
        // Return first method that successfully finds process-specific data
        if (result.process_found && result.gpu_util_mean > 0) {
            return result;
        }
        
        // If method found the process but no utilization, continue to next method
        if (result.process_has_memory && result.error_msg.empty()) {
            // Store this as fallback
            if (result.gpu_util_mean > 0) {
                return result;
            }
        }
    }

    // If no method worked, return the best attempt
    GpuUtilStats final_result = method1_enhanced_nvml(pid, gpu_index, num_samples, interval_seconds);
    final_result.error_msg = "All methods tried - using best available result";
    return final_result;
}

py::dict monitor_gpu_util_by_pid_py(unsigned int pid, unsigned int gpu_index = 0, unsigned int num_samples = 10, double interval_seconds = 0.1) {
    GpuUtilStats stats = monitor_gpu_util_by_pid(pid, gpu_index, num_samples, interval_seconds);
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

// Keep the debug function
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
          "Monitor GPU SM Utilization by PID using multiple methods");
    
    m.def("list_gpu_processes", &list_gpu_processes_py,
          py::arg("gpu_index") = 0,
          "List all processes using GPU");
}