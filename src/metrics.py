import json
import threading
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

# Thread-safe simple metrics store
_lock = threading.Lock()

_request_count = 0
_error_count = 0
_latencies = []
_prediction_labels = Counter()

LOG_FILE = Path("logs/predictions.jsonl")


def _ensure_log_dir():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)


def record_request(
    latency: float,
    success: bool = True,
    label: str = None,
    order: Dict[str, Any] = None,
    result: Dict[str, Any] = None,
):
    global _request_count, _error_count
    with _lock:
        _request_count += 1
        if not success:
            _error_count += 1
        _latencies.append(latency)
        if len(_latencies) > 1000:
            _latencies.pop(0)
        if label:
            _prediction_labels[label] += 1

        # Store prediction log (only on success)
        if success and order is not None and result is not None:
            _ensure_log_dir()
            log_entry = {
                "timestamp": datetime.utcnow().isoformat(),
                "order": order,
                "prediction": result.get("prediction"),
                "probability": result.get("probability"),
                "label": result.get("label"),
                "model_version": result.get("model_version"),
                "latency_seconds": latency,
            }
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(log_entry) + "\n")


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
