from __future__ import annotations

import uvicorn

from .settings import Settings


def main() -> None:
    settings = Settings.from_env()
    uvicorn.run(
        "kaggle_host_llm.app:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )


if __name__ == "__main__":
    main()

