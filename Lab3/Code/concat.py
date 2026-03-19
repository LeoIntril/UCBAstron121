"""
concat_watcher.py
-----------------
Watches the output directory for new sun_data_*.npz files and 
concatenates them into a single growing master file in real time.

Usage (in a separate terminal, while observe_sun.py is running):
    python3 concat_watcher.py

Output:
    sun_data_master.npz  — grows as new partial files arrive
"""

import numpy as np
import os
import time
import glob

OUTPUT_DIR  = "."
MASTER_FILE = os.path.join(OUTPUT_DIR, "sun_data_master.npz")
POLL_SEC    = 30   # how often to check for new files (seconds)

def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return d["vis"], d["times"], d["alt_az"]

def get_partial_files():
    """Return sorted list of partial files, excluding the master."""
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "sun_data_*.npz")))
    return [f for f in files if "master" not in os.path.basename(f)]

def concat_into_master():
    partial_files = get_partial_files()
    if not partial_files:
        return 0

    all_vis, all_times, all_alt_az = [], [], []

    for f in partial_files:
        try:
            vis, times, alt_az = load_npz(f)
            all_vis.append(vis)
            all_times.append(times)
            all_alt_az.append(alt_az)
        except Exception as e:
            print(f"[watcher] Could not read {f}: {e} — skipping.")

    if not all_vis:
        return 0

    vis_cat    = np.concatenate(all_vis,    axis=0)
    times_cat  = np.concatenate(all_times,  axis=0)
    alt_az_cat = np.concatenate(all_alt_az, axis=0)

    np.savez(MASTER_FILE, vis=vis_cat, times=times_cat, alt_az=alt_az_cat)
    return vis_cat.shape[0]

def main():
    print("[watcher] Starting. Watching for new .npz files...")
    seen_counts = {}  # track file -> integration count to detect genuinely new files

    while True:
        n = concat_into_master()
        if n > 0:
            print(f"[watcher] Master file updated: {n} total integrations.")
        else:
            print("[watcher] No data yet.")
        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()
