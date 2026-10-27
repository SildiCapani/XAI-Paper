"""Absolute Pfade in den smarty_workspace.

Alles dort wird nur gelesen: kein Kopieren, kein Installieren, kein Schreiben.
Über die Umgebungsvariable LANE_DETECTION_ROOT lässt sich der Ort überschreiben.
"""

import os
import sys
from pathlib import Path

LANE_DETECTION_ROOT = Path(
    os.environ.get(
        "LANE_DETECTION_ROOT",
        Path.home() / "Documents/smarty_workspace/src/lane_detection_ai",
    )
)

MODEL_DIR = LANE_DETECTION_ROOT / "models"
# Name -> (Gewichte, Config). Auswahl per --model oder XAI_MODEL.
MODELS = {
    "bev": (MODEL_DIR / "best_model_bev.pth", LANE_DETECTION_ROOT / "config/model_bev_config.py"),
    "bev_2026": (MODEL_DIR / "best_model_bev_2026.pth", MODEL_DIR / "best_model_bev_2026_config.py"),
}
DEFAULT_MODEL = os.environ.get("XAI_MODEL", "bev")
TEST_IMAGE_DIR = MODEL_DIR / "eval_2025-03-06"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = PROJECT_ROOT / "results"


def results_dir(model: str) -> Path:
    """Ergebnisse je Modell getrennt: results/<modell>/..."""
    return RESULTS_ROOT / model
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"


def add_lane_detection_to_path() -> None:
    """Macht das Paket ``lane_detection_ai`` importierbar, ohne es zu installieren."""
    root = str(LANE_DETECTION_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
