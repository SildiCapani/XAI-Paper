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
MODEL_PATH = MODEL_DIR / "best_model_bev_2026.pth"
MODEL_CONFIG_PATH = MODEL_DIR / "best_model_bev_2026_config.py"
TEST_IMAGE_DIR = MODEL_DIR / "eval_2025-03-06"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "paper" / "figures"


def add_lane_detection_to_path() -> None:
    """Macht das Paket ``lane_detection_ai`` importierbar, ohne es zu installieren."""
    root = str(LANE_DETECTION_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
