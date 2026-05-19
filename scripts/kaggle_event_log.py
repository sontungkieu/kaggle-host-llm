import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from kaggle_host_llm.kaggle_log import classify_error, log_kaggle_event, main, summarize


if __name__ == "__main__":
    main()
