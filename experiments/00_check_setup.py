"""Prüft das Setup: Modell lädt mit strict=True, Forward-Pass, Gradienten bis layer4.

Aufruf:  .venv/bin/python experiments/00_check_setup.py
"""

import torch

from xai_lane import paths
from xai_lane.model_loader import load_model
from xai_lane.preprocess import load_gray, preprocess


def main() -> None:
    net, cfg = load_model()
    print(f"Modell geladen (strict=True), Eingabe {cfg.train_width}x{cfg.train_height}")

    images = sorted(paths.TEST_IMAGE_DIR.glob("*.jpg"))
    print(f"{len(images)} Testbilder in {paths.TEST_IMAGE_DIR}")
    gray = load_gray(images[0])
    x, _ = preprocess(gray, cfg.train_width, cfg.train_height)

    feats = {}
    net.model.layer4.register_forward_hook(lambda m, i, o: feats.update(layer4=o))
    out = net(x)
    for name, t in out.items():
        print(f"  {name:10s} {tuple(t.shape)}")
    print(f"  layer4     {tuple(feats['layer4'].shape)}")

    feats["layer4"].retain_grad()
    out["exist_row"][0, 1].sum().backward()
    g = feats["layer4"].grad
    assert g is not None and g.abs().sum() > 0, "keine Gradienten an layer4"
    print("Gradienten an layer4 vorhanden – Setup ok.")


if __name__ == "__main__":
    with torch.enable_grad():
        main()
