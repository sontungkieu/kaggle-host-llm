from __future__ import annotations

import argparse
import base64
import json
import mimetypes
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen


USER_AGENT = "kaggle-host-llm-test-client/1.0"


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
        description="Send an OCR request through the gateway to verify the pipeline."
    )
    parser.add_argument("--env-file", default=".secrets/.env")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="paddleocr-ppstructurev3")
    parser.add_argument("--image-url", default="", help="Remote image/document URL.")
    parser.add_argument("--image-file", default="", help="Local image/document file.")
    parser.add_argument(
        "--return-format",
        choices=["markdown", "text", "json", "all"],
        default="markdown",
    )
    parser.add_argument("--timeout", type=float, default=600)
    return parser


def payload_from_args(args: argparse.Namespace) -> dict[str, object]:
    payload: dict[str, object] = {
        "model": args.model,
        "return_format": args.return_format,
    }
    if args.image_url:
        payload["image_url"] = args.image_url
        return payload
    if args.image_file:
        path = Path(args.image_file)
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        payload["image_base64"] = data
        payload["filename"] = path.name
        payload["mime_type"] = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return payload
    raise SystemExit("Provide --image-url or --image-file.")


def main() -> None:
    args = build_parser().parse_args()
    env = load_env(Path(args.env_file))
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": USER_AGENT,
    }
    api_key = env.get("GATEWAY_API_KEY", "")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(
        f"{args.base_url.rstrip('/')}/v1/ocr",
        data=json.dumps(payload_from_args(args)).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=args.timeout) as response:
            body = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        error_body = exc.read().decode("utf-8")
        raise SystemExit(f"Gateway returned HTTP {exc.code}: {error_body}") from exc
    print(json.dumps(body, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
