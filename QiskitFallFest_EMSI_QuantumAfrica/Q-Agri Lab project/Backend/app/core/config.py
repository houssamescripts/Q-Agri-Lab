from pathlib import Path
import os

BACKEND_DIR = Path(__file__).resolve().parents[2]
REPO_DIR = BACKEND_DIR.parent.parent
DATA_DIR = Path(os.getenv("QAGRI_DATA_DIR", REPO_DIR / "data"))
try:
    from dotenv import load_dotenv
    # The project currently uses .env.example for local credentials. Allow an
    # explicit path override; prefer that file when present, then conventional
    # .env for deployments that keep secrets outside the example file.
    env_path = Path(os.getenv("QAGRI_ENV_FILE", REPO_DIR / ".env.example"))
    if not env_path.is_absolute():
        env_path = REPO_DIR / env_path
    if not env_path.exists():
        env_path = REPO_DIR / ".env"
    load_dotenv(env_path, override=True)
except ImportError:
    pass
FEATURES = ["B1", "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12", "NDVI"]
CLASS_NAMES = ["alfalfa", "maize", "wheat"]
