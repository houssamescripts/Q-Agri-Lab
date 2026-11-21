"""Classical challenge baseline using the frozen reference configuration."""
import json

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from app.core.config import REPO_DIR


def build_svm(seed: int = 0):
    config_path = REPO_DIR / "configs" / "reference.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("name") != "rbf_svm":
        raise ValueError("Challenge reference configuration must define rbf_svm")
    return make_pipeline(StandardScaler(), SVC(random_state=seed)).set_params(**config["params"])
