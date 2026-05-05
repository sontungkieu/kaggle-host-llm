from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlsplit


TUNNEL_URL_RE = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")


def update_env_value(env_path: Path, key: str, value: str) -> None:
    lines = env_path.read_text(encoding="utf-8").splitlines() if env_path.exists() else []
    output: list[str] = []
    replaced = False
    for line in lines:
        if line.startswith(f"{key}="):
            output.append(f"{key}={value}")
            replaced = True
        else:
            output.append(line)
    if not replaced:
        output.append(f"{key}={value}")
    env_path.parent.mkdir(parents=True, exist_ok=True)
    env_path.write_text("\n".join(output) + "\n", encoding="utf-8")
    env_path.chmod(0o600)


def update_gateway_env(env_path: Path, tunnel_url: str) -> dict[str, str]:
    parsed = urlsplit(tunnel_url)
    if parsed.scheme != "https" or not parsed.netloc:
        raise ValueError(f"Unexpected tunnel URL: {tunnel_url}")

    values = {
        "GATEWAY_PUBLIC_HOSTNAME": parsed.netloc,
        "GATEWAY_PUBLIC_URL": tunnel_url,
        "GATEWAY_WS_URL": f"wss://{parsed.netloc}/workers/connect",
        "GATEWAY_HEALTH_URL": f"https://{parsed.netloc}/health",
    }
    for key, value in values.items():
        update_env_value(env_path, key, value)
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start a Cloudflare Quick Tunnel and update .secrets/.env."
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Local gateway origin URL.",
    )
    parser.add_argument(
        "--env-file",
        default=".secrets/.env",
        help=(
            "Env file to update with GATEWAY_PUBLIC_HOSTNAME, "
            "GATEWAY_PUBLIC_URL, GATEWAY_WS_URL, and GATEWAY_HEALTH_URL."
        ),
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=8,
        help="Number of Quick Tunnel creation attempts before giving up.",
    )
    parser.add_argument(
        "--retry-delay",
        type=float,
        default=4.0,
        help="Delay between failed Quick Tunnel attempts.",
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

    command = [
        cloudflared,
        "tunnel",
        "--edge-ip-version",
        "4",
        "--protocol",
        "http2",
        "--url",
        args.url,
    ]

    for attempt in range(1, args.retries + 1):
        print(f"Starting Cloudflare Quick Tunnel attempt {attempt}/{args.retries}...", flush=True)
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        tunnel_url = ""
        try:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                match = TUNNEL_URL_RE.search(line)
                if match and not tunnel_url:
                    tunnel_url = match.group(0)
                    updated_values = update_gateway_env(Path(args.env_file), tunnel_url)
                    print(
                        f"Updated {args.env_file}: "
                        f"GATEWAY_PUBLIC_HOSTNAME={updated_values['GATEWAY_PUBLIC_HOSTNAME']}",
                        flush=True,
                    )
                    print(
                        f"Updated {args.env_file}: "
                        f"GATEWAY_WS_URL={updated_values['GATEWAY_WS_URL']}",
                        flush=True,
                    )
                    print(
                        f"Updated {args.env_file}: "
                        f"GATEWAY_HEALTH_URL={updated_values['GATEWAY_HEALTH_URL']}",
                        flush=True,
                    )
                if "failed to unmarshal quick Tunnel" in line:
                    break
            exit_code = process.poll()
            if tunnel_url:
                if exit_code is None:
                    print("Tunnel is running. Keep this process open.", flush=True)
                    process.wait()
                return
            if exit_code is None:
                process.terminate()
                process.wait(timeout=5)
        except KeyboardInterrupt:
            if process.poll() is None:
                process.terminate()
            raise
        except Exception:
            if process.poll() is None:
                process.terminate()
            raise

        if attempt < args.retries:
            time.sleep(args.retry_delay)

    print("Cloudflare Quick Tunnel failed after retries.", file=sys.stderr)
    print("Use a named Cloudflare Tunnel if Quick Tunnel keeps returning 500.", file=sys.stderr)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
