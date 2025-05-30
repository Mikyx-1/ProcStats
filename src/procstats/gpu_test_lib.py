import subprocess
import time

from procstats_pid import list_gpu_processes, monitor_gpu_util_by_pid


def get_target_pid():
    """Find Python processes using GPU"""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            for line in result.stdout.strip().split("\n"):
                if line.strip() and "python" in line.lower():
                    pid = int(line.split(",")[0].strip())
                    return pid
    except:
        pass
    return None


def main():
    # Auto-detect or use manual PID
    target_pid = get_target_pid()
    if target_pid is None:
        target_pid = int(input("Enter PID to monitor: "))

    print(f"=== Enhanced GPU Process Monitoring ===")
    print(f"Target PID: {target_pid}")

    # Test with aggressive parameters for better detection
    print("\nRunning enhanced monitoring...")

    try:
        stats = monitor_gpu_util_by_pid(
            target_pid,
            gpu_index=1,  # Based on your previous results
            num_samples=50,  # More samples
            interval_seconds=0.05,  # Faster sampling
        )

        print(f"\n=== Results ===")
        print(f"Method used: {stats['method_used']}")
        print(f"Process found: {stats['process_found']}")
        print(f"Process has memory: {stats['process_has_memory']}")
        print(f"Memory used: {stats['memory_used_mb']:.1f} MB")
        print(f"Number of samples collected: {stats['num_samples']}")

        if stats["error_msg"]:
            print(f"Error/Warning: {stats['error_msg']}")

        if stats["gpu_util_mean"] > 0:
            print(f"\n🎯 SUCCESS! Process-specific GPU utilization:")
            print(f"  Mean utilization: {stats['gpu_util_mean']:.1f}%")
            print(f"  Max utilization: {stats['gpu_util_max']}%")

            # Show sample distribution
            samples = stats["samples"]
            if len(samples) > 0:
                print(f"  Sample range: {min(samples):.1f}% - {max(samples):.1f}%")
                print(
                    f"  Non-zero samples: {sum(1 for s in samples if s > 0)}/{len(samples)}"
                )
        else:
            print(f"\n⚠️  No utilization data captured")
            print("This could mean:")
            print("- Process is between GPU operations")
            print("- Very short GPU bursts")
            print("- Driver/permission limitations")

    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
