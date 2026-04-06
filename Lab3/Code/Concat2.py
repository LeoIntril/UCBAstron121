"""
sync_watcher.py
---------------
Runs on your LAPTOP. Pulls new partial .npz files from the RPi,
confirms they arrived safely, then deletes them from the RPi to
keep the RPi storage clean during long observations.

Usage:
    python3 sync_watcher.py
"""

import subprocess
import time
import os
import hashlib
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

RPI_USER     = "pi"
RPI_HOST     = "your-rpi-hostname-or-ip"
RPI_DATA_DIR = "~/your_project/data"

today        = datetime.now().strftime('%Y-%m-%d')
LOCAL_DIR    = Path(f"data/{today}")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

SYNC_INTERVAL_SEC = 60

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_rpi_partials():
    """List partial .npz files currently on the RPi."""
    cmd = [
        "ssh", f"{RPI_USER}@{RPI_HOST}",
        f"ls {RPI_DATA_DIR}/{today}/sun_data_*_partial.npz 2>/dev/null"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not result.stdout.strip():
        return []
    return [f.strip() for f in result.stdout.splitlines()]

def verify_npz(path):
    """
    Verify a transferred .npz file is valid and not corrupted
    by attempting to load it.
    """
    try:
        d = np.load(str(path), allow_pickle=False)
        # Check expected keys are present
        assert "vis" in d and "times" in d and "alt_az" in d
        # Check data is non-empty
        assert d["vis"].shape[0] > 0
        return True
    except Exception as e:
        print(f"[sync] Verification failed for {path.name}: {e}")
        return False

def rsync_file(rpi_path):
    """
    Transfer a single file from RPi to laptop using rsync.
    Returns local path if successful, None otherwise.
    """
    filename   = os.path.basename(rpi_path)
    local_path = LOCAL_DIR / filename

    # Skip if already transferred and verified
    if local_path.exists() and verify_npz(local_path):
        return local_path

    cmd = [
        "rsync",
        "-avz",
        "--checksum",       # verify transfer integrity via checksum, not just timestamp
        f"{RPI_USER}@{RPI_HOST}:{rpi_path}",
        str(local_path)
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        return local_path
    else:
        print(f"[sync] rsync failed for {filename}:\n{result.stderr}")
        return None

def delete_from_rpi(rpi_path):
    """Delete a file from the RPi after confirmed safe transfer."""
    cmd = ["ssh", f"{RPI_USER}@{RPI_HOST}", f"rm {rpi_path}"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[sync] Deleted from RPi: {os.path.basename(rpi_path)}")
    else:
        print(f"[sync] Failed to delete {os.path.basename(rpi_path)} from RPi: {result.stderr}")

def get_rpi_disk_usage():
    """Report RPi disk usage for monitoring."""
    cmd = ["ssh", f"{RPI_USER}@{RPI_HOST}", "df -h /"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        lines = result.stdout.splitlines()
        if len(lines) > 1:
            return lines[1]   # the data line
    return "unknown"

# ---------------------------------------------------------------------------
# Main sync loop
# ---------------------------------------------------------------------------

def main():
    print(f"[sync] Starting RPi data sync.")
    print(f"[sync] RPi source : {RPI_USER}@{RPI_HOST}:{RPI_DATA_DIR}/{today}/")
    print(f"[sync] Local dest : {LOCAL_DIR.resolve()}")
    print(f"[sync] Sync interval: {SYNC_INTERVAL_SEC}s\n")

    transferred_total = 0

    while True:
        rpi_files = get_rpi_partials()

        if not rpi_files:
            print("[sync] No partial files on RPi yet.")
        else:
            print(f"[sync] Found {len(rpi_files)} partial file(s) on RPi.")

            for rpi_path in rpi_files:
                filename = os.path.basename(rpi_path)

                # 1. Transfer file
                local_path = rsync_file(rpi_path)
                if local_path is None:
                    print(f"[sync] Transfer failed for {filename} — keeping on RPi.")
                    continue

                # 2. Verify transfer integrity
                if not verify_npz(local_path):
                    print(f"[sync] {filename} failed verification — keeping on RPi.")
                    continue

                # 3. Only delete from RPi after confirmed good local copy
                delete_from_rpi(rpi_path)
                transferred_total += 1
                print(f"[sync] Successfully transferred and verified: {filename}")

        # Report local file count and RPi disk usage
        local_files = list(LOCAL_DIR.glob("sun_data_*_partial.npz"))
        disk_usage  = get_rpi_disk_usage()
        print(f"[sync] {transferred_total} file(s) transferred total. "
              f"Local partial files: {len(local_files)}. "
              f"RPi disk: {disk_usage}")
        print(f"[sync] Next sync in {SYNC_INTERVAL_SEC}s...")
        time.sleep(SYNC_INTERVAL_SEC)

if __name__ == "__main__":
    main()
```

The key safety principle is the **three-step transfer sequence** for each file:
```
1. rsync with --checksum  →  transfers and verifies bytes match
2. verify_npz()           →  confirms file loads correctly in numpy
3. delete_from_rpi()      →  only runs if both steps 1 and 2 pass
```

This means a file is never deleted from the RPi unless you have a confirmed good copy locally. The `--checksum` flag on rsync is important here — it verifies the actual file contents rather than just timestamps, so a corrupted transfer will be caught before deletion.

On the laptop, `concat_watcher.py` picks up the transferred partials automatically since it's watching the same `LOCAL_DIR`. The full pipeline then looks like:
```
RPi                         Laptop
──────────────────          ────────────────────────────────
observe_sun.py              sync_watcher.py
  ↓ saves partials            ↓ transfers + verifies + deletes from RPi
  data/2026-04-01/            data/2026-04-01/
    partial files  ─rsync→     partial files
    (deleted after)              ↓
                              concat_watcher.py
                                ↓
                              sun_data_main.npz
