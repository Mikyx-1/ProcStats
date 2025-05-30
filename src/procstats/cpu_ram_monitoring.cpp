#include <pybind11/pybind11.h>
#include <pybind11/functional.h>
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

namespace py = pybind11;

struct MonitorStats {
    double cpu_max = 0.0;
    double cpu_avg = 0.0;
    double ram_max = 0.0;
    double ram_avg = 0.0;
};

// Read RAM usage in MB from /proc/[pid]/status
double get_ram_usage(pid_t pid) {
    std::string path = "/proc/" + std::to_string(pid) + "/status";
    std::ifstream file(path);
    std::string line;
    while (std::getline(file, line)) {
        if (line.find("VmRSS:") == 0) {
            std::istringstream iss(line);
            std::string key;
            double value_kb;
            std::string unit;
            iss >> key >> value_kb >> unit;
            return value_kb / 1024.0;  // Convert KB to MB
        }
    }
    return 0.0;
}

// Read user+system CPU time for the process from /proc/[pid]/stat
long get_proc_cpu_time(pid_t pid) {
    std::ifstream file("/proc/" + std::to_string(pid) + "/stat");
    std::string value;
    long utime = 0, stime = 0;
    for (int i = 1; i <= 15 && file >> value; ++i) {
        if (i == 14) utime = std::stol(value);
        if (i == 15) stime = std::stol(value);
    }
    return utime + stime;  // In clock ticks
}

// Read total CPU time from /proc/stat
long get_total_cpu_time() {
    std::ifstream file("/proc/stat");
    std::string line;
    std::getline(file, line); // first line: cpu ...
    std::istringstream iss(line);
    std::string cpu;
    long total = 0, val;
    iss >> cpu;
    while (iss >> val) {
        total += val;
    }
    return total;
}

MonitorStats monitor_process(pid_t pid, double interval_sec, double timeout_sec) {
    std::vector<double> ram_samples;
    std::vector<double> cpu_samples;

    auto start_time = std::chrono::steady_clock::now();

    long prev_proc_time = get_proc_cpu_time(pid);
    long prev_total_time = get_total_cpu_time();

    int status;
    while (true) {
        pid_t result = waitpid(pid, &status, WNOHANG);
        if (result == pid)
            break;

        // RAM
        ram_samples.push_back(get_ram_usage(pid));

        // Sleep for interval
        std::this_thread::sleep_for(std::chrono::duration<double>(interval_sec));

        // CPU
        long curr_proc_time = get_proc_cpu_time(pid);
        long curr_total_time = get_total_cpu_time();

        long delta_proc = curr_proc_time - prev_proc_time;
        long delta_total = curr_total_time - prev_total_time;

        if (delta_total > 0) {
            double cpu_usage = 100.0 * delta_proc / delta_total * std::thread::hardware_concurrency();
            cpu_samples.push_back(cpu_usage);
        }

        prev_proc_time = curr_proc_time;
        prev_total_time = curr_total_time;

        // Timeout
        auto elapsed = std::chrono::steady_clock::now() - start_time;
        if (timeout_sec > 0 && elapsed > std::chrono::duration<double>(timeout_sec)) {
            kill(pid, SIGKILL);
            break;
        }
    }

    MonitorStats stats;

    if (!ram_samples.empty()) {
        stats.ram_max = *std::max_element(ram_samples.begin(), ram_samples.end());
        stats.ram_avg = std::accumulate(ram_samples.begin(), ram_samples.end(), 0.0) / ram_samples.size();
    }

    if (!cpu_samples.empty()) {
        stats.cpu_max = *std::max_element(cpu_samples.begin(), cpu_samples.end());
        stats.cpu_avg = std::accumulate(cpu_samples.begin(), cpu_samples.end(), 0.0) / cpu_samples.size();
    }

    return stats;
}

py::dict full_resource_monitor(py::function py_func, py::object gpu_index = py::none(),
                               double timeout = 10.0, std::string monitor = "both") {
    pid_t pid = fork();
    if (pid == 0) {
        py_func();  // Run Python function in child process
        _exit(0);
    }

    MonitorStats stats = monitor_process(pid, 0.1, timeout);
    waitpid(pid, nullptr, 0);  // Ensure child fully terminated

    py::dict result;
    result["cpu_max"] = stats.cpu_max;
    result["cpu_avg"] = stats.cpu_avg;
    result["ram_max"] = stats.ram_max;
    result["ram_avg"] = stats.ram_avg;
    return result;
}

PYBIND11_MODULE(procstats, m) {
    m.def("full_resource_monitor", &full_resource_monitor,
          py::arg("target"),
          py::arg("gpu_index") = py::none(),
          py::arg("timeout") = 10.0,
          py::arg("monitor") = "both");
}
