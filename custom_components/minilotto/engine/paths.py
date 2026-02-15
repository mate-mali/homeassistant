"""
Path utility — all paths derived from a single root directory.
In HA context: /config/minilotto/
"""

from pathlib import Path


def get_paths(root: str | Path):
    """Return a dict of all directory paths rooted at *root*."""
    root = Path(root)
    paths = {
        "root": root,
        "data": root / "data",
        "models": root / "trained_models",
        "models_standard": root / "trained_models" / "standard",
        "models_weighted": root / "trained_models" / "weighted",
        "models_transition": root / "trained_models" / "transition",
        "predictions": root / "predictions",
        "predictions_standard": root / "predictions" / "standard",
        "predictions_weighted": root / "predictions" / "weighted",
        "predictions_transition": root / "predictions" / "transition",
        "predictions_mp_window": root / "predictions" / "mp_window",
        "accuracy": root / "accuracy",
    }
    for d in paths.values():
        d.mkdir(parents=True, exist_ok=True)
    return paths
