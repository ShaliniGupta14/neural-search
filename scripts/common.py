import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from app.config import RESULTS_DIR


def environment_info() -> dict[str, str]:
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": str(torch.backends.mps.is_available()),
    }


def save_json(name: str, payload: Any) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / name
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def markdown_table(rows: list[dict[str, Any]], columns: list[str], float_format: str = "{:.3f}") -> str:
    def fmt(value: Any) -> str:
        return float_format.format(value) if isinstance(value, float) else str(value)

    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    lines += ["| " + " | ".join(fmt(row.get(col, "")) for col in columns) + " |" for row in rows]
    return "\n".join(lines)
