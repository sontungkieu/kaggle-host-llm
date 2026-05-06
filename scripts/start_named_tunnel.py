from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


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


def find_tunnel_token(values: dict[str, str], explicit_key: str) -> str:
    candidate_keys = [
        explicit_key,
        "CLOUDFLARE_TUNNEL_TOKEN",
        "TUNNEL_TOKEN",
        "TOKEN_CUA_TUNNEL",
    ]
    for key in candidate_keys:
        value = values.get(key) or os.environ.get(key)
        if value:
            return value
    raise SystemExit(
        "No Cloudflare tunnel token found. Set TOKEN_CUA_TUNNEL in .secrets/.env "
        "or pass --token-env-key with the env var name to use."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start a Cloudflare named tunnel using a token from .secrets/.env."
    )
    parser.add_argument(
        "--env-file",
        default=".secrets/.env",
        help="Env file containing TOKEN_CUA_TUNNEL or CLOUDFLARE_TUNNEL_TOKEN.",
    )
    parser.add_argument(
        "--token-env-key",
        default="TOKEN_CUA_TUNNEL",
        help="Preferred env key to read the tunnel token from.",
    )
    parser.add_argument(
        "--loglevel",
        default="info",
        choices=["debug", "info", "warn", "error", "fatal"],
        help="cloudflared log level.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    cloudflared = shutil.which("cloudflared")
    if not cloudflared:
        fallback = Path.home() / ".local/bin/cloudflared"
        cloudflared = str(fallback) if fallback.exists() else ""
    if not cloudflared:
        raise SystemExit("cloudflared not found. Install it or add it to PATH.")

    env_path = Path(args.env_file)
    values = load_env_file(env_path)
    token = find_tunnel_token(values, args.token_env_key)

    command_env = os.environ.copy()
    command_env["TUNNEL_TOKEN"] = token
    command = [
        cloudflared,
        "tunnel",
        "--loglevel",
        args.loglevel,
        "run",
    ]
    print(
        f"Starting Cloudflare named tunnel from {env_path} "
        "using TUNNEL_TOKEN environment variable.",
        flush=True,
    )
    print("Keep this process open while the gateway should be public.", flush=True)
    try:
        subprocess.run(command, check=True, env=command_env)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(exc.returncode) from exc
    except KeyboardInterrupt:
        print("Named tunnel stopped.", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
