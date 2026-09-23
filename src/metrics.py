import threading
from collections import Counter
from typing import Dict

# Thread-safe simple metrics store
_lock = threading.Lock()

_request_count = 0
_error_count = 0
_latencies = []
_prediction_labels = Counter()


def record_request(latency: float, success: bool = True, label: str = None):
    global _request_count, _error_count
    with _lock:
        _request_count += 1
        if not success:
            _error_count += 1
        _latencies.append(latency)
        # Keep only the last 1000 latencies to avoid memory growth
        if len(_latencies) > 1000:
            _latencies.pop(0)
        if label:
            _prediction_labels[label] += 1


def get_metrics() -> Dict:
    with _lock:
        total = _request_count
        errors = _error_count
        error_rate = (errors / total) if total > 0 else 0.0
        avg_latency = sum(_latencies) / len(_latencies) if _latencies else 0.0

        return {
            "request_count": total,
            "error_count": errors,
            "error_rate": round(error_rate, 4),
            "avg_latency_seconds": round(avg_latency, 4),
            "prediction_distribution": dict(_prediction_labels),
        }
