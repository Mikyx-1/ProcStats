#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
#include <pybind11/stl.h>
#include <nvml.h>
#include <chrono>
#include <thread>
#include <csignal>
#include <unistd.h>
#include <sys/wait.h>
#include <fstream>
#include <sstream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <string>
#include <stdexcept>
#include <iostream>
#include <set>
#include <dirent.h>
#include <cctype>
#include <atomic>
#include <ctime>

namespace py = pybind11;

// Structures from cpu_monitoring.cpp
struct ProcessInfo {
    long utime = 0;
    long stime = 0;
    double ram_mb = 0.0;
    bool valid = false;
};

struct MonitorStats {
    double cpu_max = 0.0;
    double cpu_avg = 0.0;
    double ram_max = 0.0;
    double ram_avg = 0.0;
    size_t sample_count = 0;
    bool process_completed = false;
    int exit_code = 0;
    size_t max_process_count = 0;
    int num_cores = 0;
    std::string cpu_type;
    std::string cpu_architecture;
    double total_ram_mb = 0.0;
};

// Structure for GPU-specific information
struct GpuInfo {
    std::string name;
    std::string architecture;
    double total_vram_mb = 0.0;
    double gpu_util_mean = 0.0;
    unsigned int gpu_util_max = 0;
    bool process_found = false;
    bool process_has_memory = false;
    unsigned long long memory_used = 0;
};

// Combined monitoring stats
struct CombinedStats {
    MonitorStats cpu_stats;
    std::vector<GpuInfo> gpu_stats;
    double execution_time_seconds = 0.0;
    std::string execution_start_time;
    std::string gpu_error_msg;
};

// Function declarations from cpu_monitoring.cpp
int get_cpu_cores();
std::string get_cpu_type();
std::string get_cpu_architecture();
double get_total_ram_mb();
bool is_all_digits(const std::string& str);
bool safe_stoi(const std::string& str, int& result);
std::set<pid_t> get_all_descendants(pid_t root_pid);
ProcessInfo get_process_info(pid_t pid);
ProcessInfo get_process_tree_info(const std::set<pid_t>& pids);
long get_system_cpu_time();
double calculate_cpu_percentage(const ProcessInfo& prev, const ProcessInfo& curr,
                               long prev_system, long curr_system, int num_cores);

// GPU monitoring function for all GPUs
void background_monitor_pid(unsigned int target_pid, double interval_seconds, 
                           std::atomic<bool>& should_stop, std::vector<std::vector<double>>& gpu_samples, 
                           std::vector<unsigned long long>& max_memories, 
                           std::vector<bool>& process_founds, unsigned int device_count) {
    nvmlReturn_t nvml_result = nvmlInit_v2();
    if (nvml_result != NVML_SUCCESS) {
        std::cerr << "NVML init failed: " << nvmlErrorString(nvml_result) << std::endl;
        return;
    }

    std::vector<nvmlDevice_t> devices(device_count);
    for (unsigned int i = 0; i < device_count; ++i) {
        nvml_result = nvmlDeviceGetHandleByIndex(i, &devices[i]);
        if (nvml_result != NVML_SUCCESS) {
            std::cerr << "Failed to get device " << i << ": " << nvmlErrorString(nvml_result) << std::endl;
            nvmlShutdown();
            return;
        }
    }

    std::vector<unsigned long long> time_windows = {500, 1000, 2000, 5000, 10000};

    while (!should_stop.load()) {
        if (kill(target_pid, 0) != 0) {
            break;
        }

        for (unsigned int i = 0; i < device_count; ++i) {
            unsigned int processCount = 128;
            nvmlProcessInfo_t processes[128];
            nvml_result = nvmlDeviceGetComputeRunningProcesses(devices[i], &processCount, processes);
            
            if (nvml_result == NVML_SUCCESS) {
                for (unsigned int j = 0; j < processCount; ++j) {
                    if (processes[j].pid == target_pid) {
                        process_founds[i] = true;
                        if (processes[j].usedGpuMemory > max_memories[i]) {
                            max_memories[i] = processes[j].usedGpuMemory;
                        }
                        break;
                    }
                }
            }

            bool found_util = false;
            for (auto time_window : time_windows) {
                unsigned int sampleCount = 128;
                nvmlProcessUtilizationSample_t util_samples[128];
                
                nvml_result = nvmlDeviceGetProcessUtilization(devices[i], util_samples, &sampleCount, time_window);
                if (nvml_result == NVML_SUCCESS && sampleCount > 0) {
                    for (unsigned int j = 0; j < sampleCount; ++j) {
                        if (util_samples[j].pid == target_pid && util_samples[j].smUtil > 0) {
                            gpu_samples[i].push_back(static_cast<double>(util_samples[j].smUtil));
                            found_util = true;
                            break;
                        }
                    }
                    if (found_util) break;
                }
            }
        }

        std::this_thread::sleep_for(std::chrono::duration<double>(interval_seconds));
    }

    nvmlShutdown();
}

CombinedStats combined_monitor_process_tree(pid_t root_pid, double interval_sec, double timeout_sec) {
    CombinedStats result;
    std::vector<double> ram_samples;
    std::vector<double> cpu_samples;
    std::vector<std::vector<double>> gpu_samples;
    std::vector<unsigned long long> max_gpu_memories;
    std::vector<bool> gpu_process_founds;
    std::atomic<bool> should_stop(false);
    
    const auto start_time = std::chrono::steady_clock::now();
    auto start_time_t = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
    std::stringstream ss;
    ss << std::ctime(&start_time_t);
    result.execution_start_time = ss.str();
    result.execution_start_time.erase(result.execution_start_time.find_last_not_of("\n\r") + 1);

    const auto interval_duration = std::chrono::duration<double>(interval_sec);
    const double min_interval = 0.01;
    
    if (interval_sec < min_interval) {
        interval_sec = min_interval;
    }
    
    // Initialize CPU-related stats
    int num_cores = get_cpu_cores();
    result.cpu_stats.num_cores = num_cores;
    result.cpu_stats.cpu_type = get_cpu_type();
    result.cpu_stats.cpu_architecture = get_cpu_architecture();
    result.cpu_stats.total_ram_mb = get_total_ram_mb();
    
    // Initialize GPU-related stats
    unsigned int device_count = 0;
    bool gpu_available = true;
    nvmlReturn_t nvml_result = nvmlInit_v2();
    if (nvml_result != NVML_SUCCESS) {
        result.gpu_error_msg = "NVML init failed: " + std::string(nvmlErrorString(nvml_result));
        gpu_available = false;
    } else {
        nvml_result = nvmlDeviceGetCount(&device_count);
        if (nvml_result != NVML_SUCCESS) {
            result.gpu_error_msg = "Failed to get GPU count: " + std::string(nvmlErrorString(nvml_result));
            gpu_available = false;
            nvmlShutdown();
        }
    }

    if (gpu_available && device_count > 0) {
        result.gpu_stats.resize(device_count);
        gpu_samples.resize(device_count);
        max_gpu_memories.resize(device_count, 0);
        gpu_process_founds.resize(device_count, false);
        for (unsigned int i = 0; i < device_count; ++i) {
            nvmlDevice_t device;
            if (nvmlDeviceGetHandleByIndex(i, &device) == NVML_SUCCESS) {
                char name[NVML_DEVICE_NAME_BUFFER_SIZE];
                if (nvmlDeviceGetName(device, name, NVML_DEVICE_NAME_BUFFER_SIZE) == NVML_SUCCESS) {
                    result.gpu_stats[i].name = name;
                } else {
                    result.gpu_stats[i].name = "Unknown";
                }

                // Get GPU architecture
                int major, minor;
                if (nvmlDeviceGetCudaComputeCapability(device, &major, &minor) == NVML_SUCCESS) {
                    if (major == 8 && minor == 0) result.gpu_stats[i].architecture = "Ampere";
                    else if (major == 8 && minor == 9) result.gpu_stats[i].architecture = "Ada Lovelace";
                    else if (major == 7 && minor == 5) result.gpu_stats[i].architecture = "Turing";
                    else if (major == 7 && minor == 0) result.gpu_stats[i].architecture = "Volta";
                    else if (major == 6) result.gpu_stats[i].architecture = "Pascal";
                    else result.gpu_stats[i].architecture = "Unknown";
                } else {
                    result.gpu_stats[i].architecture = "Unknown";
                }

                // Get total VRAM
                nvmlMemory_t memory;
                if (nvmlDeviceGetMemoryInfo(device, &memory) == NVML_SUCCESS) {
                    result.gpu_stats[i].total_vram_mb = memory.total / (1024.0 * 1024.0);
                }
            }
        }
    } else {
        result.gpu_error_msg = gpu_available ? "No GPUs found" : result.gpu_error_msg;
    }

    // Start GPU monitoring thread if GPUs are available
    std::thread gpu_monitor_thread;
    if (gpu_available && device_count > 0) {
        gpu_monitor_thread = std::thread(background_monitor_pid, root_pid, interval_sec,
                                        std::ref(should_stop), std::ref(gpu_samples),
                                        std::ref(max_gpu_memories), std::ref(gpu_process_founds),
                                        device_count);
    }
    
    ProcessInfo prev_info;
    long prev_system_time = 0;
    
    try {
        prev_system_time = get_system_cpu_time();
    } catch (const std::exception& e) {
        std::cerr << "Warning: Could not read system CPU time: " << e.what() << std::endl;
    }
    
    auto next_sample_time = start_time + interval_duration;
    std::set<pid_t> current_pids = get_all_descendants(root_pid);
    result.cpu_stats.max_process_count = current_pids.size();
    
    while (true) {
        int status;
        pid_t wait_result = waitpid(root_pid, &status, WNOHANG);
        
        if (wait_result == root_pid) {
            result.cpu_stats.process_completed = true;
            if (WIFEXITED(status)) {
                result.cpu_stats.exit_code = WEXITSTATUS(status);
            } else if (WIFSIGNALED(status)) {
                result.cpu_stats.exit_code = -WTERMSIG(status);
            }
            break;
        } else if (wait_result == -1) {
            break;
        }
        
        current_pids = get_all_descendants(root_pid);
        result.cpu_stats.max_process_count = std::max(result.cpu_stats.max_process_count, current_pids.size());
        
        ProcessInfo curr_info = get_process_tree_info(current_pids);
        if (!curr_info.valid) {
            break;
        }
        
        ram_samples.push_back(curr_info.ram_mb);
        
        if (prev_info.valid) {
            try {
                long curr_system_time = get_system_cpu_time();
                double cpu_percent = calculate_cpu_percentage(
                    prev_info, curr_info, prev_system_time, curr_system_time, num_cores);
                
                if (cpu_percent >= 0) {
                    cpu_samples.push_back(cpu_percent);
                }
                
                prev_system_time = curr_system_time;
            } catch (const std::exception& e) {
                std::cerr << "Warning: CPU calculation failed: " << e.what() << std::endl;
            }
        }
        
        prev_info = curr_info;
        
        auto current_time = std::chrono::steady_clock::now();
        if (timeout_sec > 0) {
            auto elapsed = current_time - start_time;
            if (elapsed > std::chrono::duration<double>(timeout_sec)) {
                for (pid_t pid : current_pids) {
                    kill(pid, SIGTERM);
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(100));
                
                for (pid_t pid : current_pids) {
                    kill(pid, SIGKILL);
                }
                break;
            }
        }
        
        std::this_thread::sleep_until(next_sample_time);
        next_sample_time += interval_duration;
    }
    
    // Stop GPU monitoring
    should_stop.store(true);
    if (gpu_available && device_count > 0) {
        gpu_monitor_thread.join();
    }
    
    // Process CPU stats
    result.cpu_stats.sample_count = ram_samples.size();
    if (!ram_samples.empty()) {
        result.cpu_stats.ram_max = *std::max_element(ram_samples.begin(), ram_samples.end());
        result.cpu_stats.ram_avg = std::accumulate(ram_samples.begin(), ram_samples.end(), 0.0) / ram_samples.size();
    }
    
    if (!cpu_samples.empty()) {
        result.cpu_stats.cpu_max = *std::max_element(cpu_samples.begin(), cpu_samples.end());
        result.cpu_stats.cpu_avg = std::accumulate(cpu_samples.begin(), cpu_samples.end(), 0.0) / cpu_samples.size();
    }
    
    // Process GPU stats
    if (gpu_available && device_count > 0) {
        for (unsigned int i = 0; i < device_count; ++i) {
            result.gpu_stats[i].memory_used = max_gpu_memories[i];
            result.gpu_stats[i].process_has_memory = result.gpu_stats[i].memory_used > 0;
            result.gpu_stats[i].process_found = gpu_process_founds[i];
            if (!gpu_samples[i].empty()) {
                result.gpu_stats[i].gpu_util_max = static_cast<unsigned int>(*std::max_element(gpu_samples[i].begin(), gpu_samples[i].end()));
                result.gpu_stats[i].gpu_util_mean = std::accumulate(gpu_samples[i].begin(), gpu_samples[i].end(), 0.0) / gpu_samples[i].size();
            }
        }
    }
    
    result.execution_time_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - start_time).count();
    
    if (!gpu_available) {
        result.gpu_error_msg = "No GPU support available or NVML initialization failed";
    }
    
    return result;
}

py::dict combined_resource_monitor(py::function py_func, double timeout = 10.0, double interval = 0.1) {
    if (interval <= 0) {
        throw std::invalid_argument("Interval must be positive");
    }
    if (timeout < 0) {
        throw std::invalid_argument("Timeout cannot be negative");
    }
    
    pid_t pid = fork();
    if (pid == -1) {
        throw std::runtime_error("Fork failed");
    }
    
    if (pid == 0) {
        try {
            py_func();
            _exit(0);
        } catch (...) {
            _exit(1);
        }
    }
    
    CombinedStats stats = combined_monitor_process_tree(pid, interval, timeout);
    
    int final_status;
    pid_t cleanup_result = waitpid(pid, &final_status, 0);
    if (cleanup_result == -1 && !stats.cpu_stats.process_completed) {
        std::cerr << "Warning: Could not wait for child process cleanup" << std::endl;
    }
    
    py::dict result;
    result["cpu_max"] = stats.cpu_stats.cpu_max;
    result["cpu_avg"] = stats.cpu_stats.cpu_avg;
    result["ram_max"] = stats.cpu_stats.ram_max;
    result["ram_avg"] = stats.cpu_stats.ram_avg;
    result["sample_count"] = stats.cpu_stats.sample_count;
    result["process_completed"] = stats.cpu_stats.process_completed;
    result["exit_code"] = stats.cpu_stats.exit_code;
    result["max_process_count"] = stats.cpu_stats.max_process_count;
    result["num_cores"] = stats.cpu_stats.num_cores;
    result["cpu_type"] = stats.cpu_stats.cpu_type;
    result["cpu_architecture"] = stats.cpu_stats.cpu_architecture;
    result["total_ram_mb"] = stats.cpu_stats.total_ram_mb;
    
    py::list gpu_stats_list;
    for (const auto& gpu : stats.gpu_stats) {
        py::dict gpu_dict;
        gpu_dict["name"] = gpu.name;
        gpu_dict["architecture"] = gpu.architecture;
        gpu_dict["total_vram_mb"] = gpu.total_vram_mb;
        gpu_dict["gpu_util_mean"] = gpu.gpu_util_mean;
        gpu_dict["gpu_util_max"] = gpu.gpu_util_max;
        gpu_dict["process_found"] = gpu.process_found;
        gpu_dict["process_has_memory"] = gpu.process_has_memory;
        gpu_dict["memory_used_mb"] = gpu.memory_used / (1024.0 * 1024.0);
        gpu_stats_list.append(gpu_dict);
    }
    result["gpu_stats"] = gpu_stats_list;
    
    result["execution_time_seconds"] = stats.execution_time_seconds;
    result["execution_start_time"] = stats.execution_start_time;
    result["gpu_error_msg"] = stats.gpu_error_msg;
    
    return result;
}

// CPU monitoring helper functions
int get_cpu_cores() {
    static int cached_cores = 0;
    if (cached_cores > 0) {
        return cached_cores;
    }
    
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (cpuinfo.is_open()) {
        std::string line;
        int core_count = 0;
        while (std::getline(cpuinfo, line)) {
            if (line.find("processor") == 0) {
                core_count++;
            }
        }
        if (core_count > 0) {
            cached_cores = core_count;
            return cached_cores;
        }
    }
    
    long cores = sysconf(_SC_NPROCESSORS_ONLN);
    if (cores > 0) {
        cached_cores = static_cast<int>(cores);
        return cached_cores;
    }
    
    cached_cores = 1;
    return cached_cores;
}

std::string get_cpu_type() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open()) {
        return "Unknown";
    }
    
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("model name") == 0) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string cpu_type = line.substr(pos + 1);
                cpu_type.erase(0, cpu_type.find_first_not_of(" \t"));
                cpu_type.erase(cpu_type.find_last_not_of(" \t") + 1);
                return cpu_type;
            }
        }
    }
    return "Unknown";
}

std::string get_cpu_architecture() {
    std::ifstream cpuinfo("/proc/cpuinfo");
    if (!cpuinfo.is_open()) {
        return "Unknown";
    }
    
    std::string line;
    while (std::getline(cpuinfo, line)) {
        if (line.find("architecture") == 0 || line.find("cpu architecture") == 0) {
            size_t pos = line.find(':');
            if (pos != std::string::npos) {
                std::string arch = line.substr(pos + 1);
                arch.erase(0, arch.find_first_not_of(" \t"));
                arch.erase(arch.find_last_not_of(" \t") + 1);
                return arch;
            }
        }
    }
    #if defined(__x86_64__)
        return "x86_64";
    #elif defined(__i386__)
        return "i386";
    #elif defined(__arm__)
        return "arm";
    #elif defined(__aarch64__)
        return "aarch64";
    #else
        return "Unknown";
    #endif
}

double get_total_ram_mb() {
    std::ifstream meminfo("/proc/meminfo");
    if (!meminfo.is_open()) {
        return 0.0;
    }
    
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.find("MemTotal:") == 0) {
            std::istringstream iss(line);
            std::string key;
            double value_kb;
            std::string unit;
            if (iss >> key >> value_kb >> unit) {
                return value_kb / 1024.0;
            }
        }
    }
    return 0.0;
}

bool is_all_digits(const std::string& str) {
    if (str.empty()) return false;
    for (char c : str) {
        if (!std::isdigit(c)) return false;
    }
    return true;
}

bool safe_stoi(const std::string& str, int& result) {
    if (!is_all_digits(str)) return false;
    try {
        result = std::stoi(str);
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

std::set<pid_t> get_all_descendants(pid_t root_pid) {
    std::set<pid_t> descendants;
    std::vector<pid_t> to_check = {root_pid};
    
    while (!to_check.empty()) {
        pid_t current_pid = to_check.back();
        to_check.pop_back();
        
        if (descendants.find(current_pid) != descendants.end()) {
            continue;
        }
        
        descendants.insert(current_pid);
        
        DIR* proc_dir = opendir("/proc");
        if (!proc_dir) continue;
        
        struct dirent* entry;
        while ((entry = readdir(proc_dir)) != nullptr) {
            std::string dir_name = entry->d_name;
            if (!is_all_digits(dir_name)) continue;
            
            int pid_int;
            if (!safe_stoi(dir_name, pid_int)) continue;
            pid_t pid = static_cast<pid_t>(pid_int);
            
            std::string stat_path = "/proc/" + dir_name + "/stat";
            std::ifstream stat_file(stat_path);
            if (!stat_file.is_open()) continue;
            
            std::string line;
            if (!std::getline(stat_file, line)) continue;
            
            std::istringstream iss(line);
            std::string token;
            pid_t ppid = 0;
            
            std::vector<std::string> tokens;
            while (iss >> token) {
                tokens.push_back(token);
            }
            
            if (tokens.size() >= 4) {
                size_t ppid_index = 3;
                for (size_t i = 1; i < tokens.size(); ++i) {
                    if (tokens[i].back() == ')') {
                        ppid_index = i + 2;
                        break;
                    }
                }
                
                if (ppid_index < tokens.size()) {
                    try {
                        ppid = std::stoi(tokens[ppid_index]);
                    } catch (const std::exception&) {
                        continue;
                    }
                }
            }
            
            if (ppid == current_pid) {
                to_check.push_back(pid);
            }
        }
        closedir(proc_dir);
    }
    
    return descendants;
}

ProcessInfo get_process_info(pid_t pid) {
    ProcessInfo info;
    
    std::string stat_path = "/proc/" + std::to_string(pid) + "/stat";
    std::ifstream stat_file(stat_path);
    if (!stat_file.is_open()) {
        return info;
    }
    
    std::string line;
    if (!std::getline(stat_file, line)) {
        return info;
    }
    
    std::istringstream iss(line);
    std::vector<std::string> tokens;
    std::string token;
    while (iss >> token) {
        tokens.push_back(token);
    }
    
    if (tokens.size() >= 15) {
        size_t base_index = 0;
        for (size_t i = 1; i < tokens.size(); ++i) {
            if (tokens[i].back() == ')') {
                base_index = i + 1;
                break;
            }
        }
        
        if (base_index + 12 < tokens.size()) {
            try {
                info.utime = std::stol(tokens[base_index + 11]);
                info.stime = std::stol(tokens[base_index + 12]);
            } catch (const std::exception&) {
                return info;
            }
        }
    }
    
    std::string status_path = "/proc/" + std::to_string(pid) + "/status";
    std::ifstream status_file(status_path);
    if (status_file.is_open()) {
        std::string status_line;
        while (std::getline(status_file, status_line)) {
            if (status_line.find("VmRSS:") == 0) {
                std::istringstream status_iss(status_line);
                std::string key, unit;
                double value_kb;
                if (status_iss >> key >> value_kb >> unit) {
                    info.ram_mb = value_kb / 1024.0;
                }
                break;
            }
        }
    }
    
    info.valid = true;
    return info;
}

ProcessInfo get_process_tree_info(const std::set<pid_t>& pids) {
    ProcessInfo total_info;
    total_info.valid = false;
    
    bool any_valid = false;
    
    for (pid_t pid : pids) {
        ProcessInfo info = get_process_info(pid);
        if (info.valid) {
            total_info.utime += info.utime;
            total_info.stime += info.stime;
            total_info.ram_mb += info.ram_mb;
            any_valid = true;
        }
    }
    
    total_info.valid = any_valid;
    return total_info;
}

long get_system_cpu_time() {
    std::ifstream file("/proc/stat");
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open /proc/stat");
    }
    
    std::string line;
    if (!std::getline(file, line)) {
        throw std::runtime_error("Cannot read from /proc/stat");
    }
    
    std::istringstream iss(line);
    std::string cpu_label;
    iss >> cpu_label;
    
    long total = 0;
    long value;
    while (iss >> value) {
        total += value;
    }
    
    return total;
}

double calculate_cpu_percentage(const ProcessInfo& prev, const ProcessInfo& curr,
                               long prev_system, long curr_system, int num_cores) {
    if (!prev.valid || !curr.valid) {
        return 0.0;
    }
    
    long delta_proc = (curr.utime + curr.stime) - (prev.utime + prev.stime);
    long delta_system = curr_system - prev_system;
    
    if (delta_system <= 0) {
        return 0.0;
    }
    
    double cpu_percent = (100.0 * delta_proc * num_cores) / delta_system;
    return std::max(cpu_percent, 0.0);
}

PYBIND11_MODULE(combined_procstats, m) {
    m.doc() = "Combined CPU, RAM, and GPU resource monitoring with child process tracking";
    
    m.def("combined_resource_monitor", &combined_resource_monitor,
          py::arg("target"),
          py::arg("timeout") = 10.0,
          py::arg("interval") = 0.1,
          "Monitor CPU, RAM, and GPU usage including all child processes");
}