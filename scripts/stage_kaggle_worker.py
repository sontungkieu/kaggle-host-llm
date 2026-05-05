from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")[:90] or "kaggle-worker"


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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a Kaggle kernel staging folder for the worker notebook."
    )
    parser.add_argument("--owner", required=True, help="Kaggle username that owns the kernel.")
    parser.add_argument(
        "--notebook",
        default="notebooks/kaggle_qwen_worker.ipynb",
        help="Worker notebook template to copy into the staging folder.",
    )
    parser.add_argument(
        "--staging-root",
        default="kaggle_staging",
        help="Local staging root ignored by git.",
    )
    parser.add_argument(
        "--served-model",
        default="qwen2.5-9b-quantized",
        help="Model name advertised to the gateway.",
    )
    parser.add_argument(
        "--accelerator",
        default="NvidiaTeslaT4",
        help="Kaggle accelerator passed to `kaggle kernels push --accelerator`.",
    )
    parser.add_argument(
        "--title-prefix",
        default="Kaggle Qwen Worker",
        help="Human-readable Kaggle kernel title prefix.",
    )
    return parser


def stage_worker(
    *,
    owner: str,
    notebook: str | Path = "notebooks/kaggle_qwen_worker.ipynb",
    staging_root: str | Path = "kaggle_staging",
    served_model: str = "qwen2.5-9b-quantized",
    accelerator: str = "NvidiaTeslaT4",
    title_prefix: str = "Kaggle Qwen Worker",
    runtime_config: dict[str, Any] | None = None,
) -> tuple[Path, str]:
    notebook = Path(notebook)
    if not notebook.exists():
        raise SystemExit(f"Notebook template not found: {notebook}")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    base_slug = slugify(f"{notebook.stem}-{owner}-{timestamp}")
    staging_root_path = Path(staging_root)
    staging_dir = staging_root_path / base_slug
    suffix = 2
    while staging_dir.exists():
        staging_dir = staging_root_path / f"{base_slug}-{suffix}"
        suffix += 1
    staging_dir.mkdir(parents=True, exist_ok=False)
    kernel_slug = staging_dir.name

    staged_notebook = staging_dir / notebook.name
    shutil.copy2(notebook, staged_notebook)
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
    metadata_path = staging_dir / "kernel-metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    env_example = staging_dir / "worker.env.example"
    env_example.write_text(
        "\n".join(
            [
                "GATEWAY_WS_URL=wss://your-domain.example/workers/connect",
                "WORKER_TOKEN=change-me-worker-token",
                f"SERVED_MODEL={served_model}",
                "MODEL_ID=Qwen/Qwen2.5-7B-Instruct",
                "LOAD_IN_4BIT=true",
                "HF_TOKEN=",
                "MAX_WORKER_JOBS=auto",
                "KEEPALIVE_LOG_SECONDS=60",
                f"KAGGLE_ACCELERATOR={accelerator}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    if runtime_config:
        (staging_dir / "worker_config.json").write_text(
            json.dumps(runtime_config, indent=2) + "\n",
            encoding="utf-8",
        )
    return staging_dir, metadata["id"]


def main() -> None:
    args = build_parser().parse_args()
    staging_dir, kernel_id = stage_worker(
        owner=args.owner,
        notebook=args.notebook,
        staging_root=args.staging_root,
        served_model=args.served_model,
        accelerator=args.accelerator,
        title_prefix=args.title_prefix,
    )

    print(f"Created staging folder: {staging_dir}")
    print(f"Kernel id: {kernel_id}")
    print("Push with:")
    print(f"  kaggle kernels push -p {staging_dir} --accelerator {args.accelerator}")


if __name__ == "__main__":
    main()
