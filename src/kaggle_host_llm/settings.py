from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from .env_loader import load_default_env_files


@dataclass(frozen=True)
class Settings:
    host: str = "0.0.0.0"
    port: int = 8000
    database_path: str = "data/gateway.sqlite3"
    live_nodes_path: str = "data/live_workers.json"
    api_key: str = ""
    worker_token: str = ""
    heartbeat_timeout_seconds: float = 45.0
    alive_check_interval_seconds: float = 300.0
    job_timeout_seconds: float = 600.0
    groq_key_file: str = ".secrets/groq_key.env"
    groq_base_url: str = "https://api.groq.com/openai/v1"
    app_name: str = "kaggle-host-llm"

    @classmethod
    def from_env(cls) -> "Settings":
        load_default_env_files()
        return cls(
            host=os.getenv("HOST", "0.0.0.0"),
            port=int(os.getenv("PORT", "8000")),
            database_path=os.getenv("DATABASE_PATH", "data/gateway.sqlite3"),
            live_nodes_path=os.getenv("LIVE_NODES_PATH", "data/live_workers.json"),
            api_key=os.getenv("GATEWAY_API_KEY", ""),
            worker_token=os.getenv("WORKER_SHARED_TOKEN", ""),
            heartbeat_timeout_seconds=float(os.getenv("HEARTBEAT_TIMEOUT_SECONDS", "45")),
            alive_check_interval_seconds=float(os.getenv("ALIVE_CHECK_INTERVAL_SECONDS", "300")),
            job_timeout_seconds=float(os.getenv("JOB_TIMEOUT_SECONDS", "600")),
            groq_key_file=os.getenv("GROQ_KEY_FILE", ".secrets/groq_key.env"),
            groq_base_url=os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1"),
            app_name=os.getenv("APP_NAME", "kaggle-host-llm"),
        )

    def ensure_database_parent(self) -> None:
        path = Path(self.database_path)
        if path.parent and str(path.parent) != ".":
            path.parent.mkdir(parents=True, exist_ok=True)
        live_path = Path(self.live_nodes_path)
        if live_path.parent and str(live_path.parent) != ".":
            live_path.parent.mkdir(parents=True, exist_ok=True)
