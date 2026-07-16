import time
import numpy as np

from tqjcp.memory import ProcessMemoryMonitor


def test_process_memory_monitor_sees_native_allocation():
    monitor = ProcessMemoryMonitor(interval_s=0.002)
    with monitor:
        array = np.ones((8_000_000,), dtype=np.float64)
        assert float(array.sum()) == 8_000_000.0
        time.sleep(0.03)
    stats = monitor.stats
    assert stats is not None
    assert stats.rss_peak_mib >= stats.rss_start_mib
    assert stats.rss_delta_peak_mib >= 20.0
    assert stats.ru_maxrss_mib > 0.0
    assert "sampled_rss" in stats.memory_measurement_method
