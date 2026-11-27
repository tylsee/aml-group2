import ast
import os
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent


def _merge(base, over):
    out = dict(base)
    for key, value in over.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and isinstance(out.get(key), dict) else value
    return out


def _parse(text):
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return text


def load_config(run_config=None, overrides=()):
    """Shared config, then an optional run config, then `dotted.key=value` overrides."""
    cfg = yaml.safe_load((REPO / "configs" / "shared.yaml").read_text(encoding="utf-8"))
    if run_config:
        cfg = _merge(cfg, yaml.safe_load(Path(run_config).read_text(encoding="utf-8")))
    for item in overrides:
        key, text = item.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            node = node[part]
        node[leaf] = _parse(text)
    if os.environ.get("DPD_ROOT"):
        cfg["paths"]["data_root"] = os.environ["DPD_ROOT"]
    for key in ("splits_dir", "catalogue_dev", "catalogue_final", "cache_dir", "runs_dir"):
        path = Path(cfg["paths"][key])
        cfg["paths"][key] = str(path if path.is_absolute() else REPO / path)
    return cfg


def image_dir(cfg):
    return Path(cfg["paths"]["data_root"]) / cfg["paths"].get("image_subdir", "DPD_Small_dataset")
