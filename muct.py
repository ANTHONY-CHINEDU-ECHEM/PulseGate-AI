"""MUCT face database access.

MUCT (Milborrow, Morkel and Nicolls, 2010) holds 3,755 webcam photographs of
276 volunteers taken with five cameras under ten lighting set ups. It is free
for research use. Its authors ask that the photographs are not reproduced in
public documents, so this project downloads them on demand and never
redistributes them or shows them in figures.
"""
from __future__ import annotations

import re
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd

BASE_URL = "https://raw.githubusercontent.com/StephenMilborrow/muct/master/"
ARCHIVES = tuple(f"muct-{cam}-jpg-v1.tar.gz" for cam in "abcde")
NAME_PATTERN = re.compile(r"^i(\d{3})([a-z])([a-e])-([fm])(.)\.jpg$")


def download_muct(dest: str | Path, log=print) -> Path:
    """Download and unpack the five camera archives into ``dest``."""
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    jpg_dir = dest / "jpg"
    if jpg_dir.exists() and len(list(jpg_dir.glob("*.jpg"))) >= 3755:
        log(f"MUCT already present in {jpg_dir}")
        return jpg_dir
    for name in ARCHIVES:
        target = dest / name
        if not target.exists():
            log(f"downloading {name}")
            urllib.request.urlretrieve(BASE_URL + name, target)
        log(f"unpacking {name}")
        with tarfile.open(target, "r:gz") as archive:
            members = [m for m in archive.getmembers() if m.isfile() and NAME_PATTERN.match(Path(m.name).name)]
            for member in members:
                member.name = f"jpg/{Path(member.name).name}"      # flatten and refuse odd paths
            try:
                archive.extractall(dest, members=members, filter="data")
            except TypeError:        # Python releases before the extraction filters existed
                archive.extractall(dest, members=members)
    return jpg_dir


def index_muct(jpg_dir: str | Path) -> pd.DataFrame:
    """Parse the file names into subject and capture metadata."""
    rows = []
    for path in sorted(Path(jpg_dir).glob("*.jpg")):
        match = NAME_PATTERN.match(path.name)
        if not match:
            continue
        subject, lighting, camera, gender, glasses = match.groups()
        rows.append({
            "file": path.name, "subject": f"muct{subject}", "lighting": lighting, "camera": camera,
            "gender": "female" if gender == "f" else "male", "glasses": "yes" if glasses == "g" else "no",
        })
    if not rows:
        raise FileNotFoundError(f"no MUCT images found in {jpg_dir}")
    return pd.DataFrame(rows)
