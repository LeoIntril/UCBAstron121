"""
concat_watcher.py
-----------------
Watches the output directory for new sun_data_*.npz files and
concatenates them into a single growing master file in real time.
Tracks already-processed files to avoid redundant reads.
Preserves observation metadata alongside the data.

Usage (in a separate terminal, while observe_sun.py is running):
    python3 concat_watcher.py
"""

import numpy as np
import os
import time
import glob
import json
import ugradio
from pathlib import Path
from datetime import datetime

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

today      = datetime.now().strftime('%Y-%m-%d')
OUTPUT_DIR = Path(f"data/{today}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_FILE   = OUTPUT_DIR / "sun_data_master.npz"
MANIFEST_FILE = OUTPUT_DIR / "manifest.json"   # tracks processed files
POLL_SEC      = 30

# Observation metadata to embed in the master file
METADATA = {
    "observer"      : "UC Berkeley Undergrad Radio Lab",
    "site"          : "Campbell Hall Roof",
    "lat_deg"       : ugradio.nch.lat,
    "lon_deg"       : ugradio.nch.lon,
    "alt_m"         : ugradio.nch.alt,
    "lo_freq_ghz"   : 1.54,
    "sky_freq_ghz"  : 10.68,
    "fmin_ghz"      : 1.415,
    "fmax_ghz"      : 1.665,
    "n_channels"    : 1024,
    "date"          : today,
    "point_update_sec" : 15,
    "integration_sec"  : 1.0,
}

# ---------------------------------------------------------------------------
# Manifest: track which files have already been processed
# ---------------------------------------------------------------------------

def load_manifest():
    """Load the set of already-processed filenames from disk."""
    if MANIFEST_FILE.exists():
        with open(MANIFEST_FILE, "r") as f:
            return set(json.load(f))
    return set()

def save_manifest(processed):
    """Save the updated set of processed filenames to disk."""
    with open(MANIFEST_FILE, "w") as f:
        json.dump(sorted(processed), f, indent=2)

# ---------------------------------------------------------------------------
# File helpers
# ---------------------------------------------------------------------------

def get_partial_files():
    """Return sorted list of partial files, excluding master."""
    files = sorted(glob.glob(str(OUTPUT_DIR / "sun_data_*.npz")))
    return [f for f in files if "master" not in os.path.basename(f)]

def load_npz(path):
    d = np.load(path, allow_pickle=False)
    return d["vis"], d["times"], d["alt_az"]

# ---------------------------------------------------------------------------
# Load existing master if present
# ---------------------------------------------------------------------------

def load_master():
    """Load existing master file if it exists, return empty arrays otherwise."""
    if MASTER_FILE.exists():
        d = np.load(str(MASTER_FILE), allow_pickle=False)
        return d["vis"], d["times"], d["alt_az"]
    return None, None, None

# ---------------------------------------------------------------------------
# Main concat loop
# ---------------------------------------------------------------------------

def main():
    print(f"[concat] Starting. Watching: {OUTPUT_DIR.resolve()}")
    print(f"[concat] Master file: {MASTER_FILE}")

    processed = load_manifest()
    print(f"[concat] Already processed {len(processed)} file(s) from previous run.")

    while True:
        partial_files = get_partial_files()
        new_files     = [f for f in partial_files if os.path.basename(f) not in processed]

        if new_files:
            print(f"[concat] Found {len(new_files)} new file(s) to process.")

            # Load only new data
            new_vis, new_times, new_alt_az = [], [], []
            for f in new_files:
                try:
                    vis, times, alt_az = load_npz(f)
                    new_vis.append(vis)
                    new_times.append(times)
                    new_alt_az.append(alt_az)
                    processed.add(os.path.basename(f))
                    print(f"[concat] Loaded {vis.shape[0]} integrations from {os.path.basename(f)}")
                except Exception as e:
                    print(f"[concat] Could not read {f}: {e} — skipping.")

            if new_vis:
                # Load existing master and append only new data
                master_vis, master_times, master_alt_az = load_master()

                if master_vis is not None:
                    all_vis    = np.concatenate([master_vis,    np.concatenate(new_vis,    axis=0)], axis=0)
                    all_times  = np.concatenate([master_times,  np.concatenate(new_times,  axis=0)], axis=0)
                    all_alt_az = np.concatenate([master_alt_az, np.concatenate(new_alt_az, axis=0)], axis=0)
                else:
                    all_vis    = np.concatenate(new_vis,    axis=0)
                    all_times  = np.concatenate(new_times,  axis=0)
                    all_alt_az = np.concatenate(new_alt_az, axis=0)

                # Sort by time to ensure correct ordering
                order      = np.argsort(all_times)
                all_vis    = all_vis[order]
                all_times  = all_times[order]
                all_alt_az = all_alt_az[order]

                # Save master with embedded metadata
                np.savez(
                    str(MASTER_FILE),
                    vis    = all_vis,
                    times  = all_times,
                    alt_az = all_alt_az,
                    **{f"meta_{k}": np.array(v) for k, v in METADATA.items()}
                )

                # Update manifest
                save_manifest(processed)

                print(f"[concat] Master updated: {all_vis.shape[0]} total integrations, "
                      f"spanning {(all_times[-1]-all_times[0])*24*60:.1f} min.")

        else:
            print(f"[concat] No new files. Master has "
                  f"{'no data yet' if not MASTER_FILE.exists() else 'existing data'}.")

        time.sleep(POLL_SEC)

if __name__ == "__main__":
    main()
