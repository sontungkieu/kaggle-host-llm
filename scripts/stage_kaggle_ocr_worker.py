from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


OCR_BACKENDS = {
    "paddleocr-ppstructurev3": {
        "notebook": "notebooks/kaggle_paddleocr_worker.ipynb",
        "served_model": "paddleocr-ppstructurev3",
        "title": "Kaggle PaddleOCR Worker",
    },
    "deepseek-ocr2": {
        "notebook": "notebooks/kaggle_deepseek_ocr2_worker.ipynb",
        "served_model": "deepseek-ocr2",
        "title": "Kaggle DeepSeek OCR2 Worker",
    },
}


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")[:90] or "kaggle-ocr-worker"


def embed_runtime_config(notebook_path: Path, runtime_config: dict[str, Any]) -> None:
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    replacement = (
        "EMBEDDED_RUNTIME_CONFIG = "
        + json.dumps(runtime_config, indent=4, sort_keys=True)
        + "\n"
    ).splitlines(keepends=True)
    for cell in notebook.get("cells", []):
        source = cell.get("source", [])
        for index, line in enumerate(source):
            if line == "EMBEDDED_RUNTIME_CONFIG = {}\n":
                cell["source"] = source[:index] + replacement + source[index + 1 :]
                notebook_path.write_text(
                    json.dumps(notebook, indent=2) + "\n",
                    encoding="utf-8",
                )
                return
    raise RuntimeError("Could not find EMBEDDED_RUNTIME_CONFIG placeholder in notebook")


def stage_ocr_worker(
    *,
    owner: str,
    backend: str = "paddleocr-ppstructurev3",
    notebook: str | Path | None = None,
    staging_root: str | Path = "kaggle_staging",
    served_model: str | None = None,
    accelerator: str = "NvidiaTeslaT4",
    runtime_config: dict[str, Any] | None = None,
) -> tuple[Path, str]:
    backend_defaults = OCR_BACKENDS[backend]
    notebook_path = Path(notebook or backend_defaults["notebook"])
    if not notebook_path.exists():
        raise SystemExit(f"Notebook template not found: {notebook_path}")

    served_model = served_model or backend_defaults["served_model"]
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    base_slug = slugify(f"{notebook_path.stem}-{owner}-{timestamp}")
    staging_root_path = Path(staging_root)
    staging_dir = staging_root_path / base_slug
    suffix = 2
    while staging_dir.exists():
        staging_dir = staging_root_path / f"{base_slug}-{suffix}"
        suffix += 1
    staging_dir.mkdir(parents=True, exist_ok=False)
    kernel_slug = staging_dir.name

    staged_notebook = staging_dir / notebook_path.name
    shutil.copy2(notebook_path, staged_notebook)
    if runtime_config:
        embed_runtime_config(staged_notebook, runtime_config)

    metadata = {
        "id": f"{owner}/{kernel_slug}",
        "title": kernel_slug,
        "code_file": staged_notebook.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }
    (staging_dir / "kernel-metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    env_example_lines = [
        "GATEWAY_WS_URL=wss://your-domain.example/workers/connect",
        "WORKER_TOKEN=change-me-worker-token",
        f"OCR_BACKEND={backend}",
        f"OCR_MODEL={served_model}",
        "OCR_CAPACITY=1",
        "KEEPALIVE_LOG_SECONDS=60",
        f"KAGGLE_ACCELERATOR={accelerator}",
        "PADDLEOCR_DEVICE=auto",
        "PADDLEOCR_LANG=",
        "DEEPSEEK_OCR_MODEL_ID=deepseek-ai/DeepSeek-OCR-2",
        "DEEPSEEK_OCR_PROMPT=<image>\\n<|grounding|>Convert the document to markdown.",
        "DEEPSEEK_OCR_BASE_SIZE=1024",
        "DEEPSEEK_OCR_IMAGE_SIZE=768",
        "DEEPSEEK_OCR_CROP_MODE=true",
        "DEEPSEEK_OCR_DTYPE=float16",
        "HF_TOKEN=",
        "",
    ]
    (staging_dir / "worker.env.example").write_text(
        "\n".join(env_example_lines),
        encoding="utf-8",
    )
    if runtime_config:
        (staging_dir / "worker_config.json").write_text(
            json.dumps(runtime_config, indent=2) + "\n",
            encoding="utf-8",
        )
    return staging_dir, metadata["id"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Kaggle kernel staging folder for an OCR worker notebook."
    )
    parser.add_argument("--owner", required=True, help="Kaggle username that owns the kernel.")
    parser.add_argument(
        "--backend",
        choices=sorted(OCR_BACKENDS),
        default="paddleocr-ppstructurev3",
        help="OCR worker backend to stage.",
    )
    parser.add_argument("--notebook", default="", help="Override worker notebook template.")
    parser.add_argument("--staging-root", default="kaggle_staging")
    parser.add_argument("--served-model", default="", help="Gateway OCR model name.")
    parser.add_argument("--accelerator", default="NvidiaTeslaT4")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    staging_dir, kernel_id = stage_ocr_worker(
        owner=args.owner,
        backend=args.backend,
        notebook=args.notebook or None,
        staging_root=args.staging_root,
        served_model=args.served_model or None,
        accelerator=args.accelerator,
    )
    print(f"Created staging folder: {staging_dir}")
    print(f"Kernel id: {kernel_id}")
    print("Push with:")
    print(f"  kaggle kernels push -p {staging_dir} --accelerator {args.accelerator}")


if __name__ == "__main__":
    main()
