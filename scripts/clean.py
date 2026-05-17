"""Clean build artifacts. Called by `make clean` (after `make stop`)."""
import glob
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

def remove(path):
    p = ROOT / path
    try:
        if p.is_dir():
            shutil.rmtree(p, ignore_errors=True)
        else:
            p.unlink(missing_ok=True)
    except PermissionError as e:
        print(f"  SKIP (locked): {p.relative_to(ROOT)} — {e}")

# DB and generated data
remove("data/smr_pro.db")
remove("data/indices")
remove("data/models")
remove("data/processed")

# Logs — skip if still locked by a running process
for f in glob.glob(str(ROOT / "data/logs/*.log")):
    try:
        Path(f).unlink(missing_ok=True)
    except PermissionError:
        print(f"  SKIP (locked): {Path(f).relative_to(ROOT)}")

# __pycache__
for dirpath, dirnames, _ in os.walk(ROOT):
    for d in dirnames:
        if d == "__pycache__":
            shutil.rmtree(os.path.join(dirpath, d), ignore_errors=True)

print("Clean done.")
