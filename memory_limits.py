"""Choose one reproducible Gurobi allowance for the whole experiment."""

import ctypes
import math
import os
import sys
from pathlib import Path


def physical_memory():
    """Return (total, currently available) physical bytes without a dependency."""
    if sys.platform == "win32":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in (
                    "total", "available", "page_total", "page_available",
                    "virtual_total", "virtual_available", "extended",
                )
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("Cannot read physical memory; set --memory-limit-gb explicitly")
        return status.total, status.available

    if sys.platform.startswith("linux"):
        info = {}
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value = line.split(":", 1)
            info[key] = int(value.split()[0]) * 1024
        return info["MemTotal"], info.get("MemAvailable", info["MemFree"])

    try:
        page_size = os.sysconf("SC_PAGE_SIZE")
        total = os.sysconf("SC_PHYS_PAGES") * page_size
        available = os.sysconf("SC_AVPHYS_PAGES") * page_size
        return total, available
    except (AttributeError, ValueError, OSError) as error:
        raise OSError(
            "Cannot detect available RAM on this platform; set --memory-limit-gb explicitly"
        ) from error


def resolve_memory_limit(value="auto"):
    """Limits use Gurobi's decimal GB (10**9 bytes), not GiB."""
    source = "explicit"
    if value == "auto":
        total, available = physical_memory()
        if not 0 < available <= total:
            raise OSError("Invalid RAM measurement; set --memory-limit-gb explicitly")
        # Leave substantial room for Python variable dictionaries as well as
        # other applications. This is a solver allowance, not a process cap.
        value = min(0.5 * total, 0.6 * available) / 1e9
        source = "auto"
    elif value is None or value == "none":
        return {"memory_limit_gb": None, "memory_limit_source": "disabled"}
    try:
        value = float(value)
    except (ValueError, TypeError) as error:
        raise ValueError("Memory limit must be auto, none, or a positive number of GB") from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError("Memory limit must be finite and positive")
    return {"memory_limit_gb": value, "memory_limit_source": source}
