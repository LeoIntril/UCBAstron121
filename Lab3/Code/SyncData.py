"""
Must have set up ssh keys for double hop from Computer -> ugastro email -> rpi. 
Change jump_host to your ugastro email and the RPI_USER and HOST as needed.
Not fully streamlined, possible upgrades coule be detatch username from email and make it an enterable value for each fxn.
"""
# sync_watcher.py
# ----------------
# Pulls new partial .npz files from the RPi safely

import subprocess
import time
import os
import numpy as np
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

JUMP_HOST = "leo_intriligator@ugastro.berkeley.edu"
RPI_USER  = "radiopi"
RPI_HOST  = "10.32.92.205"
RPI_DATA_DIR = "/home/radiopi/Jackals_3/data"

today = datetime.now().strftime('%Y-%m-%d')
LOCAL_DIR = Path(f"data/{today}")
LOCAL_DIR.mkdir(parents=True, exist_ok=True)

SYNC_INTERVAL_SEC = 60

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_rpi_partials():
    """List partial .npz files currently on the RPi."""
    remote_cmd = f"ls '{RPI_DATA_DIR}/{today}/sun_data_*_partial.npz' 2>/dev/null"
    cmd = [
        "ssh",
        "-J", JUMP_HOST,
        f"{RPI_USER}@{RPI_HOST}",
        f"bash -c '{remote_cmd}'"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print("[sync] SSH ERROR:", result.stderr)
        return []
    if not result.stdout.strip():
        return []
    return [f.strip() for f in result.stdout.splitlines()]

def verify_npz(path):
    """Confirm a transferred .npz file loads correctly."""
    try:
        d = np.load(str(path), allow_pickle=False)
        assert "vis" in d and "times" in d and "alt_az" in d
        assert d["vis"].shape[0] > 0
        return True
    except Exception as e:
        print(f"[sync] Verification failed for {path.name}: {e}")
        return False

def rsync_file(rpi_path):
    """Transfer a single file from RPi to laptop using rsync."""
    filename = os.path.basename(rpi_path)
    local_path = LOCAL_DIR / filename

    # Skip if already transferred and verified
    if local_path.exists() and verify_npz(local_path):
        return local_path

    remote = f"{RPI_USER}@{RPI_HOST}:'{rpi_path}'"
    cmd = [
        "rsync",
        "-avz",
        "--checksum",
        "-e", f"ssh -J {JUMP_HOST}",
        remote,
        str(local_path)
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[sync] rsync failed for {filename}:\n{result.stderr}")
        return None
    return local_path

def delete_from_rpi(rpi_path):
    """Delete a file from the RPi after confirmed safe transfer."""
    cmd = ["ssh", "-J", JUMP_HOST, f"{RPI_USER}@{RPI_HOST}", f"rm '{rpi_path}'"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print(f"[sync] Deleted from RPi: {os.path.basename(rpi_path)}")
    else:
        print(f"[sync] Failed to delete {os.path.basename(rpi_path)}: {result.stderr}")

# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    print(f"[sync] Starting RPi data sync.")
    print(f"[sync] RPi source : {RPI_USER}@{RPI_HOST}:{RPI_DATA_DIR}/{today}/")
    print(f"[sync] Local dest : {LOCAL_DIR.resolve()}")
    print(f"[sync] Sync interval: {SYNC_INTERVAL_SEC}s\n")

    while True:
        rpi_files = get_rpi_partials()
        if not rpi_files:
            print("[sync] No partial files on RPi yet.")
        else:
            print(f"[sync] Found {len(rpi_files)} partial file(s) on RPi.")

            for rpi_path in rpi_files:
                local_path = rsync_file(rpi_path)
                if local_path and verify_npz(local_path):
                    delete_from_rpi(rpi_path)
                    print(f"[sync] Successfully transferred and verified: {os.path.basename(rpi_path)}")
                else:
                    print(f"[sync] Skipping deletion for {os.path.basename(rpi_path)} due to failed transfer/verification.")

        time.sleep(SYNC_INTERVAL_SEC)

if __name__ == "__main__":
    main()
