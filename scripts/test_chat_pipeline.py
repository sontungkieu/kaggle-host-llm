from __future__ import annotations

import argparse
import json
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if path.exists():
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                values[key] = value.strip().strip('"').strip("'")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a small chat request through the gateway to verify the pipeline."
    )
    parser.add_argument(
        "--env-file",
        default=".secrets/.env",
        help="Env file containing GATEWAY_API_KEY.",
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Gateway base URL.",
    )
    parser.add_argument(
        "--model",
        default="qwen2.5-9b-quantized",
        help="Model name routed by the gateway.",
    )
    parser.add_argument(
        "--question",
        default="Say OK only.",
        help="Question to send.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=16,
        help="Maximum output tokens.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=240,
        help="HTTP timeout in seconds.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    env = load_env(Path(args.env_file))
    api_key = env.get("GATEWAY_API_KEY", "")
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.question}],
        "temperature": 0,
        "max_tokens": args.max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        f"{args.base_url.rstrip('/')}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8")
        raise SystemExit(f"Gateway returned HTTP {exc.code}: {error_body}") from exc
    answer = body["choices"][0]["message"]["content"]
    print(json.dumps({"question": args.question, "answer": answer, "raw": body}, indent=2))


if __name__ == "__main__":
    main()

