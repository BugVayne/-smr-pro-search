#!/usr/bin/env python3
"""Start or stop all microservices. Cross-platform replacement for run_all.sh."""

import os
import sys
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PIDS_FILE = ROOT / ".run_pids"
LOG_DIR = ROOT / "data" / "logs"

SERVICES = [
    ("preprocessing",  "services.preprocessing.app"),
    ("retrieval",      "services.retrieval.app"),
    ("ranking",        "services.ranking.app"),
    ("postprocessing", "services.postprocessing.app"),
    ("gateway",        "services.gateway.app"),
    ("ui",             "ui.app"),
]

env = os.environ.copy()
env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")


def stop_all():
    if not PIDS_FILE.exists():
        return
    print("Stopping services...")
    for line in PIDS_FILE.read_text().splitlines():
        pid = line.strip()
        if not pid:
            continue
        try:
            if sys.platform == "win32":
                subprocess.call(
                    ["taskkill", "/PID", pid, "/F"],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                )
            else:
                os.kill(int(pid), 15)
        except Exception:
            pass
    PIDS_FILE.unlink(missing_ok=True)
    print("Done.")


def start_all():
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    pids = []
    for name, module in SERVICES:
        logfile = LOG_DIR / f"{name}.log"
        print(f"Starting {name} -> {logfile}")
        with open(logfile, "a") as log:
            proc = subprocess.Popen(
                [sys.executable, "-m", module],
                stdout=log, stderr=log,
                env=env, cwd=ROOT,
            )
        pids.append(str(proc.pid))
    PIDS_FILE.write_text("\n".join(pids) + "\n")
    print(f"\nAll services started. PIDs in {PIDS_FILE}, logs in {LOG_DIR}/")
    print("UI:      http://localhost:8080")
    print("Gateway: http://localhost:5000")
    print("\nStop with: python scripts/run_all.py stop")


if __name__ == "__main__":
    action = sys.argv[1] if len(sys.argv) > 1 else "start"
    if action == "stop":
        stop_all()
    else:
        stop_all()
        start_all()
