# Configuration module for the paper-monitoring project.
# All tunables are read from environment variables with the PM_ prefix.
# Defaults are suitable for local development; override via .env or shell env.

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- Paths ---
    project_root: Path = Path(__file__).parent.parent
    db_path: Path = project_root / "data" / "paper_monitoring.db"
    digest_output_dir: Path = project_root / "digests"
    template_dir: Path = project_root / "src" / "templates"
    log_dir: Path = project_root / "data" / "logs"

    # --- Ollama ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen2.5:7b"
    ollama_timeout: int = 300
    ollama_max_retries: int = 3

    # --- arXiv ---
    arxiv_categories: list[str] = ["cs.LG", "cs.AI", "cs.CL", "cs.CV", "stat.ML"]
    arxiv_fetch_delay: float = 3.0
    arxiv_max_results_per_category: int = 500
    arxiv_lookback_days: int = 30

    # --- HuggingFace ---
    hf_fetch_delay: float = 1.0

    # --- Pre-filter ---
    prefilter_top_n: int = 30
    prefilter_upvote_weight: float = 2.0
    prefilter_category_priorities: dict[str, int] = {
        "cs.LG": 5,
        "stat.ML": 4,
        "cs.AI": 3,
        "cs.CL": 3,
        "cs.CV": 2,
    }

    # --- Concept linking ---
    concept_match_threshold: float = 0.85

    # --- Logging ---
    log_level: str = "INFO"

    model_config = {"env_file": ".env", "env_prefix": "PM_", "extra": "ignore"}


settings = Settings()
