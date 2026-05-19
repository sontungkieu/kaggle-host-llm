from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_KAGGLE_LOG_PATH = Path("kaggle.log")
MAX_FIELD_CHARS = 2000


def log_path() -> Path:
    return Path(os.getenv("KAGGLE_LOG_PATH", str(DEFAULT_KAGGLE_LOG_PATH)))


def sanitize_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): sanitize_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [sanitize_value(item) for item in value]
    if not isinstance(value, str):
        return value

    sanitized = re.sub(r"(?i)(token=)[^&\s]+", r"\1<redacted>", value)
    sanitized = re.sub(
        r"(?i)(KAGGLE_KEY|HF_TOKEN|WORKER_TOKEN|WORKER_SHARED_TOKEN|GATEWAY_API_KEY)=\S+",
        r"\1=<redacted>",
        sanitized,
    )
    if len(sanitized) > MAX_FIELD_CHARS:
        return sanitized[:MAX_FIELD_CHARS] + "...<truncated>"
    return sanitized


def classify_error(message: str, *, returncode: int | None = None) -> str:
    lower = message.lower()
    if "savekernel" in lower and "400 client error" in lower:
        return "kaggle_save_kernel_400"
    if "paddlex[ocr]" in lower or "pp-structurev3 dependency" in lower:
        return "paddleocr_missing_paddlex_ocr_extra"
    if "set_optimization_level" in lower:
        return "paddleocr_paddle_version_incompatible"
    if "ncclcommshrink" in lower or "libtorch_cuda.so" in lower:
        return "paddleocr_torch_nccl_conflict"
    if "asyncio.run() cannot be called from a running event loop" in lower:
        return "kaggle_notebook_asyncio_event_loop"
    if "nvidiateslat4x0" in lower or ("registered as" in lower and "t4x0" in lower):
        return "kaggle_gpu_detection_failed"
    if ("addict" in lower and "missing dependency" in lower) or (
        "requires missing dependency addict" in lower
    ):
        return "deepseek_ocr_missing_addict"
    if "masked_scatter_" in lower and "same dtypes" in lower:
        return "deepseek_ocr_dtype_mismatch"
    if "cuda out of memory" in lower or "outofmemoryerror" in lower:
        return "kaggle_cuda_out_of_memory"
    if "maximum gpu session" in lower or "maximum gpu" in lower:
        return "kaggle_gpu_session_limit"
    if "permission" in lower and "kernels.get" in lower:
        return "kaggle_kernel_permission_denied"
    if "you must authenticate" in lower or "authenticate before" in lower:
        return "kaggle_auth_required"
    if "timed out waiting for an active" in lower:
        return "gateway_worker_registration_timeout"
    if "gateway health check not ready" in lower or "timed out" in lower:
        return "gateway_health_check_error"
    if "worker disconnected" in lower:
        return "kaggle_worker_disconnected"
    if "heartbeat timed out" in lower:
        return "kaggle_worker_heartbeat_timeout"
    if "failed to send" in lower:
        return "gateway_worker_send_failed"
    if "404 not found" in lower:
        return "resource_not_found"
    if returncode:
        return f"command_exit_{returncode}"
    return "unknown_error"


def _count_existing_occurrences(path: Path, record: dict[str, Any]) -> int:
    if not path.exists():
        return 0
    wanted = (
        record.get("owner"),
        record.get("event"),
        record.get("error_type"),
        record.get("backend"),
    )
    count = 0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            old = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (
            old.get("owner"),
            old.get("event"),
            old.get("error_type"),
            old.get("backend"),
        )
        if key == wanted:
            count += 1
    return count


def log_kaggle_event(
    event: str,
    *,
    level: str = "info",
    owner: str | None = None,
    action: str | None = None,
    backend: str | None = None,
    kernel_id: str | None = None,
    node_id: str | None = None,
    staging_dir: str | Path | None = None,
    accelerator: str | None = None,
    served_model: str | None = None,
    command: list[str] | None = None,
    message: str | None = None,
    returncode: int | None = None,
    error_type: str | None = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = log_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    error_type = error_type or (
        classify_error(message or "", returncode=returncode)
        if level in {"warning", "error"}
        else None
    )
    record: dict[str, Any] = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z"),
        "level": level,
        "event": event,
    }
    optional = {
        "owner": owner,
        "action": action,
        "backend": backend,
        "kernel_id": kernel_id,
        "node_id": node_id,
        "staging_dir": str(staging_dir) if staging_dir is not None else None,
        "accelerator": accelerator,
        "served_model": served_model,
        "command": command,
        "returncode": returncode,
        "error_type": error_type,
        "message": message,
        "details": details,
    }
    record.update(
        {
            key: sanitize_value(value)
            for key, value in optional.items()
            if value not in (None, "", {})
        }
    )
    if level in {"warning", "error"}:
        record["occurrence"] = _count_existing_occurrences(path, record) + 1
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    return record


def summarize(path: Path) -> list[tuple[tuple[str, str, str, str], int]]:
    counts: Counter[tuple[str, str, str, str]] = Counter()
    if not path.exists():
        return []
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        key = (
            str(record.get("owner") or "<unknown>"),
            str(record.get("backend") or "<none>"),
            str(record.get("event") or "<none>"),
            str(record.get("error_type") or "<none>"),
        )
        counts[key] += 1
    return sorted(counts.items(), key=lambda item: (-item[1], item[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize local Kaggle event logs.")
    parser.add_argument("--path", default=str(log_path()))
    parser.add_argument("--tail", type=int, default=0, help="Print the last N raw JSONL events.")
    args = parser.parse_args()
    path = Path(args.path)
    if args.tail:
        if not path.exists():
            return
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()[-args.tail:]
        print("\n".join(lines))
        return
    for (owner, backend, event, error_type), count in summarize(path):
        print(f"{count:4d} owner={owner} backend={backend} event={event} error_type={error_type}")
