"""
System Information Module
Handles cross-platform system information detection
"""

import logging
import multiprocessing
import os
import platform
import subprocess
import sys
from typing import Dict, Optional

import psutil

logger = logging.getLogger(__name__)


class SystemInfo:
    """Cross-platform system information collector"""

    def __init__(self):
        self._cpu_info = None
        self._system_info = None

    @property
    def cpu_info(self) -> Dict[str, any]:
        """Get CPU information (cached)"""
        if self._cpu_info is None:
            self._cpu_info = self._detect_cpu_info()
        return self._cpu_info

    @property
    def system_info(self) -> Dict[str, any]:
        """Get system information (cached)"""
        if self._system_info is None:
            self._system_info = self._detect_system_info()
        return self._system_info

    def _detect_cpu_info(self) -> Dict[str, any]:
        """Detect CPU information across platforms"""
        info = {
            "name": "Unknown CPU",
            "cores": multiprocessing.cpu_count(),
            "architecture": platform.machine(),
            "frequency": None,
        }

        try:
            # Get CPU frequency
            cpu_freq = psutil.cpu_freq()
            if cpu_freq:
                info["frequency"] = cpu_freq.max
        except Exception as e:
            logger.warning(f"CPU frequency detection failed: {e}")

        # Platform-specific CPU name detection
        try:
            if sys.platform == "win32":
                info["name"] = self._get_windows_cpu_name()
            elif sys.platform == "darwin":
                info["name"] = self._get_macos_cpu_name()
            else:  # Linux and other Unix-like systems
                info["name"] = self._get_linux_cpu_name()
        except Exception as e:
            logger.warning(f"CPU name detection failed: {e}")

        return info

    def _get_windows_cpu_name(self) -> str:
        """Get CPU name on Windows"""
        try:
            import winreg

            key = winreg.OpenKey(
                winreg.HKEY_LOCAL_MACHINE,
                r"HARDWARE\DESCRIPTION\System\CentralProcessor\0",
            )
            cpu_name = winreg.QueryValueEx(key, "ProcessorNameString")[0]
            winreg.CloseKey(key)
            return cpu_name.strip()
        except Exception:
            # Fallback to WMI if available
            try:
                result = subprocess.run(
                    ["wmic", "cpu", "get", "name", "/value"],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                for line in result.stdout.split("\n"):
                    if line.startswith("Name="):
                        return line.split("=", 1)[1].strip()
            except Exception:
                pass
        return "Unknown Windows CPU"

    def _get_macos_cpu_name(self) -> str:
        """Get CPU name on macOS"""
        try:
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "Unknown macOS CPU"

    def _get_linux_cpu_name(self) -> str:
        """Get CPU name on Linux"""
        try:
            with open("/proc/cpuinfo", "r") as f:
                for line in f:
                    if line.startswith("model name"):
                        return line.split(":", 1)[1].strip()
        except Exception:
            pass
        return "Unknown Linux CPU"

    def _detect_system_info(self) -> Dict[str, any]:
        """Detect general system information"""
        info = {
            "platform": sys.platform,
            "os_name": platform.system(),
            "os_version": platform.release(),
            "total_ram_mb": 0.0,
            "python_version": platform.python_version(),
        }

        try:
            # Get total RAM
            virtual_memory = psutil.virtual_memory()
            info["total_ram_mb"] = virtual_memory.total / (1024 * 1024)
        except Exception as e:
            logger.warning(f"RAM detection failed: {e}")

        return info

    def get_all_info(self) -> Dict[str, any]:
        """Get all system information"""
        return {"cpu": self.cpu_info, "system": self.system_info}


if __name__ == "__main__":
    # Global instance
    system_info = SystemInfo()
    print(system_info.get_all_info())
