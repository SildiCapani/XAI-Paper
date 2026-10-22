"""Lädt das 2026-BEV-Modell (UFLDv2, ResNet-"9") streng mit strict=True.

Der Checkpoint ist ein reines state_dict mit Präfix "model." (Lightning-Modul).
Das Präfix wird entfernt; danach müssen alle Schlüssel exakt passen.
Der model_wrapper.py des Teams wird bewusst nicht benutzt (erwartet ck["model"]
und lädt mit strict=False, Fehler würden dort unbemerkt bleiben).
"""

from pathlib import Path

import torch

from xai_lane import paths

PREFIX = "model."


def load_config(config_path: str | Path = paths.MODEL_CONFIG_PATH):
    paths.add_lane_detection_to_path()
    from lane_detection_ai.model.utils.common import get_config

    return get_config(str(config_path))


def load_model(
    model_path: str | Path = paths.MODEL_PATH,
    config_path: str | Path = paths.MODEL_CONFIG_PATH,
):
    """Gibt (Modell im eval-Modus auf der CPU, Config) zurück."""
    cfg = load_config(config_path)
    from lane_detection_ai.model.model.model_smartrollerz import get_model

    net = get_model(cfg)
    state = torch.load(str(model_path), map_location="cpu", weights_only=True)
    if not all(k.startswith(PREFIX) for k in state):
        raise ValueError(f"Unerwartetes Checkpoint-Format: nicht alle Schlüssel beginnen mit {PREFIX!r}")
    state = {k[len(PREFIX):]: v for k, v in state.items()}
    net.load_state_dict(state, strict=True)
    net.eval()
    return net, cfg
