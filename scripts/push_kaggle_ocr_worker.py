from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.request import urlopen
import json

from kaggle_event_log import log_kaggle_event
from push_kaggle_worker import (
    DEFAULT_KAGGLE_ACCOUNTS,
    DEFAULT_SECRETS_ENV,
    KAGGLE_CODE_URL_RE,
    derive_health_url,
    kaggle_command,
    load_env_file,
    load_kaggle_accounts,
    require_gateway_url,
)
from stage_kaggle_ocr_worker import OCR_BACKENDS, stage_ocr_worker


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
    backend: str = "",
    kernel_id: str = "",
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            if gateway_has_worker(health_url, owner, served_model):
                log_kaggle_event(
                    "ocr_worker_registered",
                    owner=owner,
                    backend=backend,
                    kernel_id=kernel_id,
                    served_model=served_model,
                    details={"health_url": health_url},
                )
                print("Gateway reports the Kaggle OCR worker as active.", flush=True)
                return
        except Exception as exc:
            log_kaggle_event(
                "gateway_health_check_failed",
                level="warning",
                owner=owner,
                backend=backend,
                kernel_id=kernel_id,
                served_model=served_model,
                message=repr(exc),
                details={"health_url": health_url},
            )
            print(f"Gateway health check not ready: {exc}", flush=True)
        print(
            f"Waiting for OCR worker registration at {health_url} "
            f"({poll_interval_seconds}s)...",
            flush=True,
        )
        time.sleep(poll_interval_seconds)
    message = "Timed out waiting for an active OCR worker. Check the Kaggle notebook logs."
    log_kaggle_event(
        "ocr_worker_registration_timeout",
        level="error",
        owner=owner,
        backend=backend,
        kernel_id=kernel_id,
        served_model=served_model,
        message=message,
        details={
            "health_url": health_url,
            "timeout_seconds": timeout_seconds,
            "poll_interval_seconds": poll_interval_seconds,
        },
    )
    raise SystemExit(message)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage and push a Kaggle OCR worker notebook with a selected account."
    )
    parser.add_argument("owner", help="Kaggle username to push as.")
    parser.add_argument(
        "--backend",
        choices=sorted(OCR_BACKENDS),
        default="paddleocr-ppstructurev3",
        help="OCR backend notebook to push.",
    )
    parser.add_argument("--accounts-file", default=str(DEFAULT_KAGGLE_ACCOUNTS))
    parser.add_argument("--env-file", default=str(DEFAULT_SECRETS_ENV))
    parser.add_argument("--notebook", default="", help="Override OCR worker notebook.")
    parser.add_argument("--staging-root", default="kaggle_staging")
    parser.add_argument("--served-model", default="", help="Gateway OCR model name.")
    parser.add_argument("--accelerator", default="")
    parser.add_argument("--gateway-ws-url", default="")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-active", action="store_true")
    parser.add_argument("--gateway-health-url", default="")
    parser.add_argument("--wait-timeout", type=int, default=1800)
    parser.add_argument("--poll-interval", type=int, default=30)
    return parser


def runtime_config_from_env(
    *,
    args: argparse.Namespace,
    env_values: dict[str, str],
    served_model: str,
    accelerator: str,
) -> dict[str, Any]:
    gateway_ws_url = args.gateway_ws_url or env_values.get("GATEWAY_WS_URL", "")
    return {
        "GATEWAY_WS_URL": require_gateway_url(gateway_ws_url),
        "WORKER_TOKEN": env_values.get("WORKER_SHARED_TOKEN", ""),
        "OWNER": args.owner,
        "OCR_BACKEND": args.backend,
        "OCR_MODEL": served_model,
        "OCR_CAPACITY": int(env_values.get("OCR_CAPACITY", "1")),
        "KEEPALIVE_LOG_SECONDS": int(env_values.get("KEEPALIVE_LOG_SECONDS", "60")),
        "KAGGLE_ACCELERATOR": accelerator,
        "PADDLEOCR_DEVICE": env_values.get("PADDLEOCR_DEVICE", "auto"),
        "PADDLEOCR_LANG": env_values.get("PADDLEOCR_LANG", ""),
        "DEEPSEEK_OCR_MODEL_ID": env_values.get(
            "DEEPSEEK_OCR_MODEL_ID",
            "deepseek-ai/DeepSeek-OCR-2",
        ),
        "DEEPSEEK_OCR_PROMPT": env_values.get(
            "DEEPSEEK_OCR_PROMPT",
            "<image>\n<|grounding|>Convert the document to markdown.",
        ),
        "DEEPSEEK_OCR_BASE_SIZE": int(env_values.get("DEEPSEEK_OCR_BASE_SIZE", "1024")),
        "DEEPSEEK_OCR_IMAGE_SIZE": int(env_values.get("DEEPSEEK_OCR_IMAGE_SIZE", "768")),
        "DEEPSEEK_OCR_CROP_MODE": env_values.get("DEEPSEEK_OCR_CROP_MODE", "true"),
        "DEEPSEEK_OCR_DTYPE": env_values.get("DEEPSEEK_OCR_DTYPE", "float16"),
        "HF_TOKEN": env_values.get("HF_TOKEN", ""),
    }


def main() -> None:
    args = build_parser().parse_args()
    env_values = load_env_file(Path(args.env_file))
    accounts = load_kaggle_accounts(Path(args.accounts_file))
    credential = accounts.get(args.owner)
    if credential is None:
        available = ", ".join(sorted(accounts)) or "<none>"
        log_kaggle_event(
            "account_not_found",
            level="error",
            owner=args.owner,
            backend=args.backend,
            message=f"Account {args.owner!r} not found.",
            details={"available_accounts": sorted(accounts)},
        )
        raise SystemExit(f"Account {args.owner!r} not found. Available: {available}")

    backend_defaults = OCR_BACKENDS[args.backend]
    served_model = args.served_model or env_values.get(
        "OCR_MODEL",
        backend_defaults["served_model"],
    )
    accelerator = args.accelerator or env_values.get("KAGGLE_ACCELERATOR", "NvidiaTeslaT4")
    runtime_config = runtime_config_from_env(
        args=args,
        env_values=env_values,
        served_model=served_model,
        accelerator=accelerator,
    )
    staging_dir, kernel_id = stage_ocr_worker(
        owner=args.owner,
        backend=args.backend,
        notebook=args.notebook or None,
        staging_root=args.staging_root,
        served_model=served_model,
        accelerator=accelerator,
        runtime_config=runtime_config,
    )
    print(f"Created staging folder: {staging_dir}", flush=True)
    print(f"Kernel id: {kernel_id}", flush=True)
    log_kaggle_event(
        "ocr_kernel_staged",
        owner=args.owner,
        backend=args.backend,
        kernel_id=kernel_id,
        staging_dir=staging_dir,
        accelerator=accelerator,
        served_model=served_model,
    )

    if args.dry_run:
        print("Dry run: skipped kaggle kernels push/status.", flush=True)
        log_kaggle_event(
            "ocr_dry_run",
            owner=args.owner,
            backend=args.backend,
            kernel_id=kernel_id,
            staging_dir=staging_dir,
            accelerator=accelerator,
            served_model=served_model,
        )
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
        log_kaggle_event(
            "ocr_kernel_push_started",
            owner=args.owner,
            backend=args.backend,
            kernel_id=kernel_id,
            staging_dir=staging_dir,
            accelerator=accelerator,
            served_model=served_model,
            command=push_cmd,
        )
        try:
            push_result = subprocess.run(
                push_cmd,
                check=True,
                env=command_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            if exc.stdout:
                print(exc.stdout, end="", flush=True)
            log_kaggle_event(
                "ocr_kernel_push_failed",
                level="error",
                owner=args.owner,
                backend=args.backend,
                kernel_id=kernel_id,
                staging_dir=staging_dir,
                accelerator=accelerator,
                served_model=served_model,
                command=push_cmd,
                message=exc.stdout or str(exc),
                returncode=exc.returncode,
            )
            raise SystemExit(f"Kaggle push failed with exit code {exc.returncode}.") from exc
        print(push_result.stdout, end="", flush=True)
        actual_kernel_id = kernel_id
        match = KAGGLE_CODE_URL_RE.search(push_result.stdout)
        if match:
            actual_kernel_id = f"{match.group(1)}/{match.group(2)}"
            if actual_kernel_id != kernel_id:
                print(f"Kaggle resolved kernel id: {actual_kernel_id}", flush=True)
        log_kaggle_event(
            "ocr_kernel_push_succeeded",
            owner=args.owner,
            backend=args.backend,
            kernel_id=actual_kernel_id,
            staging_dir=staging_dir,
            accelerator=accelerator,
            served_model=served_model,
            command=push_cmd,
            message=push_result.stdout,
        )
        status_cmd = [*kaggle_command(), "kernels", "status", actual_kernel_id]
        log_kaggle_event(
            "ocr_kernel_status_started",
            owner=args.owner,
            backend=args.backend,
            kernel_id=actual_kernel_id,
            accelerator=accelerator,
            served_model=served_model,
            command=status_cmd,
        )
        try:
            status_result = subprocess.run(
                status_cmd,
                check=True,
                env=command_env,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        except subprocess.CalledProcessError as exc:
            if exc.stdout:
                print(exc.stdout, end="", flush=True)
            log_kaggle_event(
                "ocr_kernel_status_failed",
                level="error",
                owner=args.owner,
                backend=args.backend,
                kernel_id=actual_kernel_id,
                accelerator=accelerator,
                served_model=served_model,
                command=status_cmd,
                message=exc.stdout or str(exc),
                returncode=exc.returncode,
            )
            raise
        print(status_result.stdout, end="", flush=True)
        log_kaggle_event(
            "ocr_kernel_status_succeeded",
            owner=args.owner,
            backend=args.backend,
            kernel_id=actual_kernel_id,
            accelerator=accelerator,
            served_model=served_model,
            command=status_cmd,
            message=status_result.stdout,
        )
        if args.wait_active:
            gateway_ws_url = args.gateway_ws_url or env_values.get("GATEWAY_WS_URL", "")
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
                backend=args.backend,
                kernel_id=actual_kernel_id,
            )


if __name__ == "__main__":
    main()
