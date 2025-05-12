import numpy as np
import psutil
import os
def monitor_memory():
    """监控内存使用情况"""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return f"内存使用: {mem_info.rss / (1024 * 1024):.2f} MB"