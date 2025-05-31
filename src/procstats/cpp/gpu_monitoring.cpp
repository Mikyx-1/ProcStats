#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>
#include <nvml.h>
#include <vector>
#include <algorithm>
#include <numeric>
#include <thread>
#include <chrono>
#include <iostream>
#include <atomic>
#include <future>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

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
    double execution_time_seconds = 0.0;
    py::object function_result;
    bool function_success = false;
    std::string function_error = "";
    unsigned int monitored_pid = 0;
};

// Enhanced NVML monitoring (same as your original)
GpuUtilStats method1_enhanced_nvml(unsigned int pid, unsigned int gpu_index, unsigned int num_samples, double interval_seconds) {
    GpuUtilStats result;
    result.method_used = "Enhanced NVML";
    result.monitored_pid = pid;
    
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

// Background monitoring function for a specific PID
void background_monitor_pid(unsigned int target_pid, unsigned int gpu_index, double interval_seconds, 
                           std::atomic<bool>& should_stop, std::vector<double>& samples, 
                           std::atomic<unsigned long long>& max_memory, std::atomic<bool>& process_found) {
    nvmlReturn_t nvml_result = nvmlInit_v2();
    if (nvml_result != NVML_SUCCESS) {
        return;
    }

    nvmlDevice_t device;
    nvml_result = nvmlDeviceGetHandleByIndex(gpu_index, &device);
    if (nvml_result != NVML_SUCCESS) {
        nvmlShutdown();
        return;
    }

    std::vector<unsigned long long> time_windows = {500, 1000, 2000, 5000, 10000};

    while (!should_stop.load()) {
        // Check if target process still exists
        if (kill(target_pid, 0) != 0) {
            // Process no longer exists
            break;
        }

        // Check memory usage
        unsigned int processCount = 128;
        nvmlProcessInfo_t processes[128];
        nvml_result = nvmlDeviceGetComputeRunningProcesses(device, &processCount, processes);
        
        if (nvml_result == NVML_SUCCESS) {
            for (unsigned int i = 0; i < processCount; ++i) {
                if (processes[i].pid == target_pid) {
                    process_found.store(true);
                    unsigned long long current_memory = max_memory.load();
                    if (processes[i].usedGpuMemory > current_memory) {
                        max_memory.store(processes[i].usedGpuMemory);
                    }
                    break;
                }
            }
        }

        // Check utilization
        bool found_util = false;
        for (auto time_window : time_windows) {
            unsigned int sampleCount = 128;
            nvmlProcessUtilizationSample_t util_samples[128];
            
            nvml_result = nvmlDeviceGetProcessUtilization(device, util_samples, &sampleCount, time_window);
            if (nvml_result == NVML_SUCCESS && sampleCount > 0) {
                for (unsigned int j = 0; j < sampleCount; ++j) {
                    if (util_samples[j].pid == target_pid && util_samples[j].smUtil > 0) {
                        samples.push_back(static_cast<double>(util_samples[j].smUtil));
                        found_util = true;
                        break;
                    }
                }
                if (found_util) break;
            }
        }

        std::this_thread::sleep_for(std::chrono::duration<double>(interval_seconds));
    }

    nvmlShutdown();
}

// Function to execute Python function in a separate process and monitor it
GpuUtilStats run_and_monitor_gpu_on_function_impl(py::function target_func, unsigned int gpu_index, double interval_seconds) {
    GpuUtilStats result;
    result.method_used = "Separate Process GPU Monitoring";
    
    // Serialize the function and its dependencies
    py::module pickle = py::module::import("pickle");
    py::module cloudpickle;
    
    try {
        cloudpickle = py::module::import("cloudpickle");
    } catch (...) {
        result.error_msg = "cloudpickle not available. Please install with: pip install cloudpickle";
        return result;
    }
    
    py::bytes serialized_func;
    try {
        serialized_func = cloudpickle.attr("dumps")(target_func);
    } catch (const std::exception& e) {
        result.error_msg = "Failed to serialize function: " + std::string(e.what());
        return result;
    }
    
    // Create pipes for communication
    int pipe_fd[2];
    if (pipe(pipe_fd) == -1) {
        result.error_msg = "Failed to create pipe";
        return result;
    }
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Fork a new process
    pid_t child_pid = fork();
    
    if (child_pid == -1) {
        result.error_msg = "Failed to fork process";
        close(pipe_fd[0]);
        close(pipe_fd[1]);
        return result;
    }
    
    if (child_pid == 0) {
        // Child process - execute the function
        close(pipe_fd[0]); // Close read end
        
        try {
            // Deserialize and execute function
            py::object func = cloudpickle.attr("loads")(serialized_func);
            py::object func_result = func();
            
            // Serialize result
            py::bytes result_bytes = pickle.attr("dumps")(py::make_tuple("success", func_result));
            std::string result_str = result_bytes.cast<std::string>();
            
            // Send result through pipe
            write(pipe_fd[1], result_str.c_str(), result_str.length());
            
        } catch (const std::exception& e) {
            // Send error through pipe
            py::bytes error_bytes = pickle.attr("dumps")(py::make_tuple("error", std::string(e.what())));
            std::string error_str = error_bytes.cast<std::string>();
            write(pipe_fd[1], error_str.c_str(), error_str.length());
        }
        
        close(pipe_fd[1]);
        _exit(0); // Exit child process
    }
    
    // Parent process - monitor the child
    close(pipe_fd[1]); // Close write end
    result.monitored_pid = child_pid;
    
    // Start monitoring thread
    std::atomic<bool> should_stop(false);
    std::atomic<bool> process_found(false);
    std::vector<double> utilization_samples;
    std::atomic<unsigned long long> max_memory_used(0);
    
    std::thread monitor_thread(background_monitor_pid, child_pid, gpu_index, interval_seconds,
                              std::ref(should_stop), std::ref(utilization_samples),
                              std::ref(max_memory_used), std::ref(process_found));
    
    // Wait for child process to complete
    int status;
    waitpid(child_pid, &status, 0);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    result.execution_time_seconds = std::chrono::duration<double>(end_time - start_time).count();
    
    // Stop monitoring
    should_stop.store(true);
    monitor_thread.join();
    
    // Read result from pipe
    char buffer[65536];
    ssize_t bytes_read = read(pipe_fd[0], buffer, sizeof(buffer) - 1);
    close(pipe_fd[0]);
    
    if (bytes_read > 0) {
        buffer[bytes_read] = '\0';
        try {
            py::bytes result_bytes(buffer, bytes_read);
            py::tuple result_tuple = pickle.attr("loads")(result_bytes);
            std::string status_str = result_tuple[0].cast<std::string>();
            
            if (status_str == "success") {
                result.function_result = result_tuple[1];
                result.function_success = true;
            } else {
                result.function_error = result_tuple[1].cast<std::string>();
                result.function_success = false;
            }
        } catch (const std::exception& e) {
            result.function_error = "Failed to deserialize result: " + std::string(e.what());
            result.function_success = false;
        }
    } else {
        result.function_error = "No result received from child process";
        result.function_success = false;
    }
    
    // Process monitoring results
    result.memory_used = max_memory_used.load();
    result.process_has_memory = result.memory_used > 0;
    result.process_found = process_found.load();
    
    if (!utilization_samples.empty()) {
        result.utilization_samples = utilization_samples;
        result.gpu_util_max = static_cast<unsigned int>(*std::max_element(utilization_samples.begin(), utilization_samples.end()));
        result.gpu_util_mean = std::accumulate(utilization_samples.begin(), utilization_samples.end(), 0.0) / utilization_samples.size();
    }
    
    return result;
}

// Original monitor function (keeping for compatibility)
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
    result["monitored_pid"] = stats.monitored_pid;
    
    py::list samples_list;
    for (auto sample : stats.utilization_samples) {
        samples_list.append(sample);
    }
    result["samples"] = samples_list;
    
    return result;
}

// New function wrapper for Python
py::dict run_and_monitor_gpu_on_function_py(py::function target_func, unsigned int gpu_index = 0, double interval = 0.01) {
    GpuUtilStats stats = run_and_monitor_gpu_on_function_impl(target_func, gpu_index, interval);
    
    py::dict result;
    result["gpu_util_mean"] = stats.gpu_util_mean;
    result["gpu_util_max"] = stats.gpu_util_max;
    result["process_found"] = stats.process_found;
    result["process_has_memory"] = stats.process_has_memory;
    result["memory_used_mb"] = stats.memory_used / (1024 * 1024);
    result["error_msg"] = stats.error_msg;
    result["method_used"] = stats.method_used;
    result["num_samples"] = static_cast<int>(stats.utilization_samples.size());
    result["execution_time_seconds"] = stats.execution_time_seconds;
    result["function_success"] = stats.function_success;
    result["function_error"] = stats.function_error;
    result["monitored_pid"] = stats.monitored_pid;
    
    if (stats.function_success) {
        result["function_result"] = stats.function_result;
    } else {
        result["function_result"] = py::none();
    }
    
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

    m.def("run_and_monitor_gpu_on_function", &run_and_monitor_gpu_on_function_py,
          py::arg("target"), py::arg("gpu_index") = 0, py::arg("interval") = 0.01,
          "Execute a Python function in separate process while monitoring GPU utilization");

    m.def("list_gpu_processes", &list_gpu_processes_py,
          py::arg("gpu_index") = 0,
          "List all processes using GPU");
}