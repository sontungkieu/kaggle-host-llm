from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Iterable

from kaggle_event_log import log_kaggle_event
from push_kaggle_worker import (
    DEFAULT_KAGGLE_ACCOUNTS,
    kaggle_command,
    load_kaggle_accounts,
)


def run_kaggle(
    command: list[str],
    *,
    command_env: dict[str, str],
    capture: bool = False,
    owner: str = "",
    action: str = "",
    kernel: str = "",
    details: dict[str, object] | None = None,
) -> subprocess.CompletedProcess[str]:
    log_kaggle_event(
        "kernel_manage_started",
        owner=owner,
        action=action,
        kernel_id=kernel,
        command=command,
        details=details,
    )
    try:
        result = subprocess.run(
            command,
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
            "kernel_manage_failed",
            level="error",
            owner=owner,
            action=action,
            kernel_id=kernel,
            command=command,
            message=exc.stdout or str(exc),
            returncode=exc.returncode,
            details=details,
        )
        raise
    if not capture and result.stdout:
        print(result.stdout, end="", flush=True)
    log_kaggle_event(
        "kernel_manage_succeeded",
        owner=owner,
        action=action,
        kernel_id=kernel,
        command=command,
        message=result.stdout,
        details=details,
    )
    return result


def parse_kernel_refs(output: str, *, owner: str, prefix: str) -> list[str]:
    lines = output.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if line.lower().startswith("ref,") or ",ref," in line.lower()
        ),
        None,
    )
    if header_index is None:
        return []

    refs: list[str] = []
    reader = csv.DictReader(lines[header_index:])
    for row in reader:
        ref = (row.get("ref") or row.get("Ref") or "").strip()
        if not ref or "/" not in ref:
            continue
        ref_owner, slug = ref.split("/", 1)
        if ref_owner == owner and slug.startswith(prefix):
            refs.append(ref)
    return refs


def list_worker_kernel_refs(
    *,
    owner: str,
    prefix: str,
    command_env: dict[str, str],
    page_size: int,
) -> list[str]:
    command = [
        *kaggle_command(),
        "kernels",
        "list",
        "--mine",
        "--sort-by",
        "dateRun",
        "--page-size",
        str(page_size),
        "--csv",
    ]
    result = run_kaggle(
        command,
        command_env=command_env,
        capture=True,
        owner=owner,
        action="list_for_delete_workers",
        details={"prefix": prefix, "page_size": page_size},
    )
    return parse_kernel_refs(result.stdout or "", owner=owner, prefix=prefix)


def selected_owners(requested_owner: str, owners: Iterable[str]) -> list[str]:
    if requested_owner == "all":
        return sorted(owners)
    return [requested_owner]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Kaggle kernel management commands with credentials from .secrets."
    )
    parser.add_argument(
        "owner",
        help="Kaggle username whose API key should be used, or 'all' for bulk worker cleanup.",
    )
    parser.add_argument(
        "action",
        choices=["list", "status", "logs", "delete", "delete-workers"],
        help="Kernel action to run.",
    )
    parser.add_argument(
        "kernel",
        nargs="?",
        help="Kernel id in owner/slug format. Required for status/logs/delete.",
    )
    parser.add_argument(
        "--accounts-file",
        default=str(DEFAULT_KAGGLE_ACCOUNTS),
        help="JSON/JSONL file containing Kaggle username/key records.",
    )
    parser.add_argument(
        "--search",
        default="kaggle-qwen-worker",
        help="Search term for list action.",
    )
    parser.add_argument(
        "--prefix",
        default="kaggle-qwen-worker",
        help="Kernel slug prefix for delete-workers.",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=200,
        help="Maximum kernels to inspect per account for delete-workers.",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Confirm delete without prompting.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    accounts = load_kaggle_accounts(Path(args.accounts_file))
    owners = selected_owners(args.owner, accounts.keys())
    if args.action != "delete-workers" and args.owner == "all":
        raise SystemExit("owner='all' is only supported for delete-workers.")
    missing_owners = [owner for owner in owners if owner not in accounts]
    if missing_owners:
        available = ", ".join(sorted(accounts)) or "<none>"
        for owner in missing_owners:
            log_kaggle_event(
                "account_not_found",
                level="error",
                owner=owner,
                action=args.action,
                message=f"Account {owner!r} not found.",
                details={"available_accounts": sorted(accounts)},
            )
        raise SystemExit(f"Account(s) not found: {', '.join(missing_owners)}. Available: {available}")
    if args.action in {"status", "logs", "delete"} and not args.kernel:
        raise SystemExit(f"{args.action} requires a kernel id, for example {args.owner}/slug")

    for owner in owners:
        credential = accounts[owner]
        run_action(args, owner=owner, credential=credential)


def run_action(args: argparse.Namespace, *, owner: str, credential: dict[str, str]) -> None:
    with tempfile.TemporaryDirectory(prefix=f"kaggle-config-{owner}-") as config_dir:
        config_path = Path(config_dir) / "kaggle.json"
        config_path.write_text(json.dumps(credential) + "\n", encoding="utf-8")
        config_path.chmod(0o600)
        command_env = os.environ.copy()
        command_env["KAGGLE_CONFIG_DIR"] = config_dir

        if args.action == "list":
            command = [
                *kaggle_command(),
                "kernels",
                "list",
                "--mine",
                "--search",
                args.search,
                "--sort-by",
                "dateRun",
            ]
            run_kaggle(
                command,
                command_env=command_env,
                owner=owner,
                action=args.action,
                details={"search": args.search},
            )
        elif args.action == "status":
            command = [*kaggle_command(), "kernels", "status", args.kernel]
            run_kaggle(
                command,
                command_env=command_env,
                owner=owner,
                action=args.action,
                kernel=args.kernel or "",
            )
        elif args.action == "logs":
            command = [*kaggle_command(), "kernels", "logs", args.kernel]
            run_kaggle(
                command,
                command_env=command_env,
                owner=owner,
                action=args.action,
                kernel=args.kernel or "",
            )
        elif args.action == "delete":
            command = [*kaggle_command(), "kernels", "delete", args.kernel]
            if args.yes:
                command.append("-y")
            run_kaggle(
                command,
                command_env=command_env,
                owner=owner,
                action=args.action,
                kernel=args.kernel or "",
                details={"confirmed": args.yes},
            )
        else:
            refs = list_worker_kernel_refs(
                owner=owner,
                prefix=args.prefix,
                command_env=command_env,
                page_size=args.page_size,
            )
            if not refs:
                print(f"{owner}: no worker kernels matched prefix {args.prefix!r}", flush=True)
                log_kaggle_event(
                    "delete_workers_no_matches",
                    owner=owner,
                    action=args.action,
                    details={"prefix": args.prefix, "page_size": args.page_size},
                )
                return
            print(f"{owner}: matched {len(refs)} worker kernel(s)", flush=True)
            log_kaggle_event(
                "delete_workers_matched",
                owner=owner,
                action=args.action,
                details={"prefix": args.prefix, "matched_refs": refs, "confirmed": args.yes},
            )
            for ref in refs:
                print(f"  {ref}", flush=True)
            if not args.yes:
                print(
                    f"{owner}: dry-run only. Re-run with --yes to delete these kernels.",
                    flush=True,
                )
                return
            for ref in refs:
                print(f"{owner}: deleting {ref}", flush=True)
                run_kaggle(
                    [*kaggle_command(), "kernels", "delete", ref, "-y"],
                    command_env=command_env,
                    owner=owner,
                    action="delete-workers",
                    kernel=ref,
                    details={"prefix": args.prefix},
                )


if __name__ == "__main__":
    main()
