"""Index of crops recorded with the collect command. Free of training dependencies."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

CLASS_NAMES = ("live", "print_matte", "print_glossy", "replay_phone", "replay_monitor", "cutout_mask")


def load_captures(capture_dir: str | Path) -> pd.DataFrame:
    """Index ``capture_dir/<label>/*.jpg`` as a manifest style table with absolute paths."""
    capture_dir = Path(capture_dir)
    rows = []
    for label_dir in sorted(p for p in capture_dir.glob("*") if p.is_dir()):
        if label_dir.name not in CLASS_NAMES:
            continue
        for path in sorted(label_dir.glob("*.jpg")):
            rows.append({
                "sample_id": f"capture_{label_dir.name}_{path.stem}", "path": str(path.resolve()), "label": int(label_dir.name == "live"),
                "species": label_dir.name, "subject": "capture", "session": path.stem.rsplit("_", 1)[0],
            })
    return pd.DataFrame(rows)
