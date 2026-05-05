from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from urllib.request import urlopen

from stage_kaggle_worker import stage_worker


DEFAULT_SECRETS_ENV = Path(".secrets/.env")
DEFAULT_KAGGLE_ACCOUNTS = Path(".secrets/all-kaggle.json")
KAGGLE_CODE_URL_RE = re.compile(r"https://www\\.kaggle\\.com/code/([^\\s/]+)/([^\\s]+)")


def load_env_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def load_kaggle_accounts(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        raise SystemExit(f"Kaggle account file not found: {path}")
    text = path.read_text(encoding="utf-8")
    accounts: dict[str, dict[str, str]] = {}

    try:
        parsed = json.loads(text)
        records = parsed.values() if isinstance(parsed, dict) else parsed
    except json.JSONDecodeError:
        records = []
        for line in text.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            records.append(json.loads(line))

    for record in records:
        if not isinstance(record, dict):
            continue
        username = str(record.get("username") or record.get("KAGGLE_USERNAME") or "")
        key = str(record.get("key") or record.get("KAGGLE_KEY") or "")
        if username and key:
            accounts[username] = {"username": username, "key": key}
    return accounts


def kaggle_command() -> list[str]:
    executable = shutil.which("kaggle")
    if executable:
        return [executable]
    uvx = shutil.which("uvx")
    if uvx:
        return [uvx, "--from", "kaggle", "kaggle"]
    uv = shutil.which("uv")
    if uv:
        return [uv, "tool", "run", "--from", "kaggle", "kaggle"]
    raise SystemExit("Neither kaggle CLI nor uv/uvx is available in PATH.")


def require_gateway_url(value: str) -> str:
    if (
        not value
        or "your-domain.example" in value
        or not (value.startswith("wss://") or value.startswith("ws://"))
    ):
        raise SystemExit(
            "Set GATEWAY_WS_URL in .secrets/.env to a WebSocket URL, "
            "for example wss://example.trycloudflare.com/workers/connect "
            "or ws://PUBLIC_IP:8000/workers/connect"
        )
    return value


def derive_health_url(gateway_ws_url: str) -> str:
    parsed = urlsplit(gateway_ws_url)
    scheme = "https" if parsed.scheme == "wss" else "http"
    return urlunsplit((scheme, parsed.netloc, "/health", "", ""))


def gateway_has_worker(health_url: str, owner: str, served_model: str) -> bool:
    with urlopen(health_url, timeout=8) as response:
        payload = json.loads(response.read().decode("utf-8"))
    workers = payload.get("workers") or []
    for worker in workers:
        if (
            worker.get("status") == "active"
            and worker.get("model") == served_model
            and worker.get("owner") == owner
        ):
            return True
    return False


def wait_for_active_worker(
    *,
    health_url: str,
    owner: str,
    served_model: str,
    timeout_seconds: int,
    poll_interval_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            if gateway_has_worker(health_url, owner, served_model):
                print("Gateway reports the Kaggle worker as active.", flush=True)
                return
        except Exception as exc:
            print(f"Gateway health check not ready: {exc}", flush=True)
        print(
            f"Waiting for worker registration at {health_url} "
            f"({poll_interval_seconds}s)...",
            flush=True,
        )
        time.sleep(poll_interval_seconds)
    raise SystemExit(
        "Timed out waiting for an active worker. Check the Kaggle notebook logs."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage and push the Kaggle worker notebook with a selected account."
    )
    parser.add_argument("owner", help="Kaggle username to push as.")
    parser.add_argument(
        "--accounts-file",
        default=str(DEFAULT_KAGGLE_ACCOUNTS),
        help="JSON/JSONL file containing Kaggle username/key records.",
    )
    parser.add_argument(
        "--env-file",
        default=str(DEFAULT_SECRETS_ENV),
        help="Local env file containing gateway and worker settings.",
    )
    parser.add_argument(
        "--notebook",
        default="notebooks/kaggle_qwen_worker.ipynb",
        help="Worker notebook template.",
    )
    parser.add_argument(
        "--staging-root",
        default="kaggle_staging",
        help="Local staging root ignored by git.",
    )
    parser.add_argument(
        "--served-model",
        default="",
        help="Gateway model name. Defaults to SERVED_MODEL from env file.",
    )
    parser.add_argument(
        "--model-id",
        default="",
        help="Actual model id loaded inside the notebook. Defaults to MODEL_ID from env file.",
    )
    parser.add_argument(
        "--accelerator",
        default="",
        help="Kaggle accelerator for this run. Defaults to KAGGLE_ACCELERATOR or NvidiaTeslaT4.",
    )
    parser.add_argument(
        "--gateway-ws-url",
        default="",
        help="Public worker WebSocket URL. Defaults to GATEWAY_WS_URL from env file.",
    )
    parser.add_argument(
        "--no-runtime-config",
        action="store_true",
        help="Do not create worker_config.json in the staging folder.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Create staging and show the kernel id without calling Kaggle.",
    )
    parser.add_argument(
        "--wait-active",
        action="store_true",
        help="After push, keep polling gateway /health until this worker registers.",
    )
    parser.add_argument(
        "--gateway-health-url",
        default="",
        help="Gateway health URL. Defaults to deriving /health from GATEWAY_WS_URL.",
    )
    parser.add_argument(
        "--wait-timeout",
        type=int,
        default=1800,
        help="Seconds to wait for worker registration when --wait-active is set.",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=30,
        help="Polling interval in seconds for --wait-active.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    env_values = load_env_file(Path(args.env_file))
    accounts = load_kaggle_accounts(Path(args.accounts_file))
    credential = accounts.get(args.owner)
    if credential is None:
        available = ", ".join(sorted(accounts)) or "<none>"
        raise SystemExit(f"Account {args.owner!r} not found. Available: {available}")

    gateway_ws_url = args.gateway_ws_url or env_values.get("GATEWAY_WS_URL", "")
    served_model = args.served_model or env_values.get("SERVED_MODEL", "qwen2.5-9b-quantized")
    model_id = args.model_id or env_values.get("MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
    accelerator = args.accelerator or env_values.get("KAGGLE_ACCELERATOR", "NvidiaTeslaT4")
    worker_token = env_values.get("WORKER_SHARED_TOKEN", "")

    runtime_config: dict[str, Any] | None = None
    if not args.no_runtime_config:
        runtime_config = {
            "GATEWAY_WS_URL": require_gateway_url(gateway_ws_url),
            "WORKER_TOKEN": worker_token,
            "OWNER": args.owner,
            "SERVED_MODEL": served_model,
            "MODEL_ID": model_id,
            "LOAD_IN_4BIT": env_values.get("LOAD_IN_4BIT", "true"),
            "HF_TOKEN": env_values.get("HF_TOKEN", ""),
            "MAX_WORKER_JOBS": env_values.get("MAX_WORKER_JOBS", "auto"),
            "KEEPALIVE_LOG_SECONDS": int(env_values.get("KEEPALIVE_LOG_SECONDS", "60")),
            "KAGGLE_ACCELERATOR": accelerator,
        }

    staging_dir, kernel_id = stage_worker(
        owner=args.owner,
        notebook=args.notebook,
        staging_root=args.staging_root,
        served_model=served_model,
        accelerator=accelerator,
        runtime_config=runtime_config,
    )
    print(f"Created staging folder: {staging_dir}", flush=True)
    print(f"Kernel id: {kernel_id}", flush=True)

    if args.dry_run:
        print("Dry run: skipped kaggle kernels push/status.", flush=True)
        return

    with tempfile.TemporaryDirectory(prefix="kaggle-config-") as config_dir:
        config_path = Path(config_dir) / "kaggle.json"
        config_path.write_text(json.dumps(credential) + "\n", encoding="utf-8")
        config_path.chmod(0o600)

        command_env = os.environ.copy()
        command_env["KAGGLE_CONFIG_DIR"] = config_dir
        push_cmd = [
            *kaggle_command(),
            "kernels",
            "push",
            "-p",
            str(staging_dir),
            "--accelerator",
            accelerator,
        ]
        push_result = subprocess.run(
            push_cmd,
            check=True,
            env=command_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(push_result.stdout, end="", flush=True)
        if "Kernel push error:" in push_result.stdout:
            raise SystemExit(
                "Kaggle rejected the push. If the message says the maximum GPU "
                "session count was reached, delete or stop old worker kernels, "
                "then push again."
            )
        actual_kernel_id = kernel_id
        match = KAGGLE_CODE_URL_RE.search(push_result.stdout)
        if match:
            actual_kernel_id = f"{match.group(1)}/{match.group(2)}"
            if actual_kernel_id != kernel_id:
                print(f"Kaggle resolved kernel id: {actual_kernel_id}", flush=True)
        status_cmd = [*kaggle_command(), "kernels", "status", actual_kernel_id]
        subprocess.run(status_cmd, check=True, env=command_env)
        if args.wait_active:
            health_url = (
                args.gateway_health_url
                or env_values.get("GATEWAY_HEALTH_URL", "")
                or derive_health_url(gateway_ws_url)
            )
            wait_for_active_worker(
                health_url=health_url,
                owner=args.owner,
                served_model=served_model,
                timeout_seconds=args.wait_timeout,
                poll_interval_seconds=args.poll_interval,
            )


if __name__ == "__main__":
    main()
