"""Lädt ein UFLDv2-Modell des Teams streng mit strict=True.

Zwei Checkpoint-Formate kommen vor:
  bev       -- {"model": state_dict, "optimizer": ...} (altes Trainingsskript)
  bev_2026  -- reines state_dict mit Präfix "model." (Lightning-Modul); das
               Präfix wird entfernt.
Danach müssen alle Schlüssel exakt passen. Der model_wrapper.py des Teams wird
bewusst nicht benutzt (lädt mit strict=False, Fehler würden dort unbemerkt bleiben).
"""

from pathlib import Path

import torch

from xai_lane import paths

LIGHTNING_PREFIX = "model."


def load_config(config_path: str | Path):
    paths.add_lane_detection_to_path()
    from lane_detection_ai.model.utils.common import get_config

    return get_config(str(config_path))


def _state_dict(ck: dict) -> dict:
    if "model" in ck and isinstance(ck["model"], dict):
        return ck["model"]
    if all(k.startswith(LIGHTNING_PREFIX) for k in ck):
        return {k[len(LIGHTNING_PREFIX):]: v for k, v in ck.items()}
    raise ValueError("Unbekanntes Checkpoint-Format")


def load_model(name: str = paths.DEFAULT_MODEL):
    """Gibt (Modell im eval-Modus auf der CPU, Config) zurück."""
    model_path, config_path = paths.MODELS[name]
    cfg = load_config(config_path)
    from lane_detection_ai.model.model.model_smartrollerz import get_model

    net = get_model(cfg)
    # weights_only=True verhindert beliebigen Code im Pickle; der Optimizer-Zustand lädt damit auch
    ck = torch.load(str(model_path), map_location="cpu", weights_only=True)
    net.load_state_dict(_state_dict(ck), strict=True)
    net.eval()
    return net, cfg
