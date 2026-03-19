"""
waterfall_watcher.py
--------------------
Watches for sun_data_*.npz files and displays a live-updating waterfall
plot of the visibility spectra (amplitude vs frequency vs time).

Usage (in a separate terminal, while observe_sun.py is running):
    python3 waterfall_watcher.py

Requires:
    pip install matplotlib numpy
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import glob
import os
import time
from pathlib import Path
from datetime import datetime

today = datetime.now().strftime('%Y-%m-%d')
OUTPUT_DIR = f"data/{today}"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OUTPUT_DIR   = "."
POLL_SEC     = 15       # how often to refresh the plot (seconds)
SNAP_SRATE   = 500e6    # SNAP sample rate in Hz (500 MHz for X-band)
N_CHANNELS   = 1024     # number of frequency channels
FMIN_GHZ = 1.415        # X-band low edge (GHz) — adjust to your LO setup
FMAX_GHZ = 1.665        # X-band high edge (GHz) — adjust to your LO setup
VMIN_DB      = None     # colorscale min in dB (None = auto)
VMAX_DB      = None     # colorscale max in dB (None = auto)

# ---------------------------------------------------------------------------

def get_partial_files():
    files = sorted(glob.glob(os.path.join(OUTPUT_DIR, "sun_data_*.npz")))
    return [f for f in files if "master" not in os.path.basename(f)]

def load_all_data():
    """Load and concatenate all partial npz files."""
    files = get_partial_files()
    if not files:
        return None, None

    all_vis, all_times = [], []
    for f in files:
        try:
            d = np.load(f, allow_pickle=False)
            all_vis.append(d["vis"])
            all_times.append(d["times"])
        except Exception as e:
            print(f"[waterfall] Could not read {f}: {e} — skipping.")

    if not all_vis:
        return None, None

    vis   = np.concatenate(all_vis,   axis=0)
    times = np.concatenate(all_times, axis=0)

    # Sort by time in case files arrive out of order
    order = np.argsort(times)
    return vis[order], times[order]

def vis_to_db(vis):
    """Convert complex visibilities to amplitude in dB."""
    amp = np.abs(vis)
    amp = np.where(amp == 0, 1e-10, amp)   # avoid log(0)
    return 20 * np.log10(amp)

def times_to_minutes(times):
    """Convert JD array to minutes elapsed since first integration."""
    return (times - times[0]) * 24 * 60

def make_freq_axis():
    return np.linspace(FMIN_GHZ, FMAX_GHZ, N_CHANNELS)

# ---------------------------------------------------------------------------

def main():
    print("[waterfall] Starting live waterfall watcher...")
    plt.ion()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7),
                             gridspec_kw={"width_ratios": [3, 1]})
    fig.suptitle("X-Band Interferometer — Live Waterfall", fontsize=13)
    fig.patch.set_facecolor("#0f0f0f")
    for ax in axes:
        ax.set_facecolor("#0f0f0f")

    ax_wfall, ax_spec = axes
    freq_axis = make_freq_axis()

    while True:
        vis, times = load_all_data()

        if vis is None or vis.shape[0] < 2:
            print("[waterfall] Waiting for data...")
            plt.pause(POLL_SEC)
            continue

        db = vis_to_db(vis)                      # shape: (N_int, N_chan)
        t_min = times_to_minutes(times)          # shape: (N_int,)

        # --- Left panel: waterfall ---
        ax_wfall.cla()
        ax_wfall.set_facecolor("#0f0f0f")

        im = ax_wfall.imshow(
            db,
            aspect="auto",
            origin="lower",
            extent=[freq_axis[0], freq_axis[-1], t_min[0], t_min[-1]],
            cmap="inferno",
            vmin=VMIN_DB if VMIN_DB else np.percentile(db, 2),
            vmax=VMAX_DB if VMAX_DB else np.percentile(db, 98),
            interpolation="nearest",
        )
        ax_wfall.set_xlabel("Frequency (GHz)", color="white")
        ax_wfall.set_ylabel("Time Elapsed (min)", color="white")
        ax_wfall.set_title("Waterfall", color="white")
        ax_wfall.tick_params(colors="white")
        for spine in ax_wfall.spines.values():
            spine.set_edgecolor("white")

        cbar = fig.colorbar(im, ax=ax_wfall, pad=0.02)
        cbar.set_label("Amplitude (dB)", color="white")
        cbar.ax.yaxis.set_tick_params(color="white")
        plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")

        # --- Right panel: latest spectrum ---
        ax_spec.cla()
        ax_spec.set_facecolor("#0f0f0f")
        ax_spec.plot(db[-1], freq_axis, color="#ff6b35", linewidth=0.8)
        ax_spec.set_xlabel("Amplitude (dB)", color="white")
        ax_spec.set_title("Latest Spectrum", color="white", fontsize=9)
        ax_spec.tick_params(colors="white")
        ax_spec.set_ylim(freq_axis[0], freq_axis[-1])
        for spine in ax_spec.spines.values():
            spine.set_edgecolor("white")

        n_int = vis.shape[0]
        elapsed = t_min[-1]
        fig.suptitle(
            f"X-Band Interferometer — Live Waterfall  |  "
            f"{n_int} integrations  |  {elapsed:.1f} min elapsed",
            color="white", fontsize=11
        )

        plt.tight_layout()
        plt.pause(POLL_SEC)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
