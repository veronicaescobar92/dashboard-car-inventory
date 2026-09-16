import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = Path(os.environ.get("REPUESTOS_DB", ROOT / "data" / "demo.sqlite"))
MODEL_PATH = Path(os.environ.get("REPUESTOS_MODEL", ROOT / "artifacts" / "model.joblib"))
