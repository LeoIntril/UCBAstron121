"""
observe_sun.py
--------------
Observation of the Sun with the X-band interferometer.
Uses a background thread to collect and buffer visibility data while the
main thread handles pointing updates.

Usage (on the RPi):
    python3 observe_sun.py

Output:
    sun_data_<timestamp>.npz  containing:
        - vis:      array of complex visibility spectra, shape (N_integrations, 1024)
        - times:    array of JD timestamps for each integration
        - alt_az:   array of (alt, az) pointings, shape (N_integrations, 2)
"""

import numpy as np
import threading
import time
import os
from datetime import datetime

import ugradio
from ugradio import interf, coord
import snap_spec.snap as snap

from pathlib import Path

import subprocess
import sys

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Observing site (Campbell Hall roof, UC Berkeley)
LAT  =  ugradio.nch.lat   # degrees N
LON  =  ugradio.nch.lon  # degrees E
ALT  =  123.0     # meters above sea level

# Observation duration and pointing update cadence
POINT_UPDATE_SEC   = 15     # re-point every 30 seconds

# Minimum elevation to observe (degrees)
MIN_ALT = 5.0

# Output directory
today = datetime.now().strftime('%Y-%m-%d')
OUTPUT_DIR = Path(f"data/{today}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Shared data buffer (written by collector thread, read by main thread)
# ---------------------------------------------------------------------------

data_buffer = {
    "vis":    [],   # complex visibility spectra
    "times":  [],   # JD of each integration
    "alt_az": [],   # (alt, az) at time of integration
}
buffer_lock = threading.Lock()
stop_event  = threading.Event()

# ---------------------------------------------------------------------------
# Data collection thread
# ---------------------------------------------------------------------------

def collect_data(spec):
    """
    Background thread: reads visibility spectra from the SNAP correlator,
    skipping duplicate integrations by comparing acc_cnt.
    Runs until stop_event is set.
    """
    print("[collector] Starting data collection thread.")
    prev_acc_cnt = None
    first_read   = True

    while not stop_event.is_set():
        try:
            data = spec.read_data()

            # On the very first read, print keys so we can verify field names
            if first_read:
                print(f"[collector] read_data() keys: {list(data.keys())}")
                first_read = False

            # Skip if this is the same integration we already saved
            acc_cnt = data.get("acc_cnt", None)
            if acc_cnt is not None and acc_cnt == prev_acc_cnt:
                time.sleep(0.05)
                continue
            prev_acc_cnt = acc_cnt

            jd_now = ugradio.timing.julian_date()

            # 'corr01' confirmed as the correct key from SNAP output
            vis_spectrum = data["corr01"]

            with buffer_lock:
                data_buffer["vis"].append(vis_spectrum)
                data_buffer["times"].append(jd_now)
                data_buffer["alt_az"].append(current_altaz)
            time.sleep(1)

        except AssertionError:
            # SNAP integration advanced mid-read — harmless, just retry
            continue
        except Exception as e:
            print(f"[collector] Unexpected error: {e}")
            raise

    print("[collector] Stopping.")

# ---------------------------------------------------------------------------
# Pointing helper
# ---------------------------------------------------------------------------

# Shared variable so the collector thread can log current pointing
current_altaz = (np.nan, np.nan)

def point_to_sun(ifm):
    """
    Compute the current alt/az of the Sun and point the interferometer there.
    sunpos() already returns coordinates in the current epoch, so no
    precession step is needed.
    Returns (alt, az) if successful, or None if the Sun is too low.
    """
    global current_altaz

    jd = ugradio.timing.julian_date()

    # sunpos returns RA in hours, Dec in degrees, current epoch
    ra, dec = coord.sunpos(jd)

    # get_altaz expects RA in hours, Dec in degrees; equinox matches sunpos output
    alt, az = coord.get_altaz(ra, dec, jd, LAT, LON, ALT, equinox='J2000')

    if alt < MIN_ALT:
        print(f"[pointing] Sun too low: alt={alt:.1f}° — holding current position.")
        return None

    print(f"[pointing] Pointing to Sun: alt={alt:.1f}°, az={az:.1f}°")
    current_altaz = (alt, az)
    ifm.point(alt, az)
    return alt, az

# ---------------------------------------------------------------------------
# Save buffer to disk - modifed to clear buffer so next save only contains new data.
# ---------------------------------------------------------------------------

def save_data(tag=""):
    """Flush the shared buffer to a .npz file, then clear saved entries."""
    with buffer_lock:
        vis    = np.array(data_buffer["vis"])
        times  = np.array(data_buffer["times"])
        alt_az = np.array(data_buffer["alt_az"])

        # Clear the buffer so the next save only contains new data
        data_buffer["vis"].clear()
        data_buffer["times"].clear()
        data_buffer["alt_az"].clear()

    if vis.shape[0] == 0:
        print("[save] No data to save.")
        return

    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    fname = os.path.join(OUTPUT_DIR, f"sun_data_{timestamp}{tag}.npz")
    np.savez(fname, vis=vis, times=times, alt_az=alt_az)
    print(f"[save] Saved {vis.shape[0]} integrations to {fname}")
    return fname

# -----------------------------
# Command Menu
# -----------------------------
def command_menu():
    """
    Runs in a separate thread, accepting user commands while
    observation continues in the background.
    """
    print("\n[menu] Command menu active.")
    watchers = {}  # track launched subprocesses
    
    while not stop_event.is_set():
        try:
            cmd = input("\nCommands: save / concat / waterfall / quit : ").strip().lower()
            
            if cmd == "save":
                save_data(tag="_manual")
            
            elif cmd == "concat":
                if "concat" in watchers and watchers["concat"].poll() is None:
                    print("[menu] Concat watcher is already running.")
                else:
                    watchers["concat"] = subprocess.Popen(
                        [sys.executable, "concat_watcher.py"],
                        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32"
                        else 0,
                    )
                    print("[menu] Concat watcher launched.")

            elif cmd == "waterfall":
                if "waterfall" in watchers and watchers["waterfall"].poll() is None:
                    print("[menu] Waterfall watcher is already running.")
                else:
                    watchers["waterfall"] = subprocess.Popen(
                        [sys.executable, "waterfall_watcher.py"],
                        creationflags=subprocess.CREATE_NEW_CONSOLE if sys.platform == "win32"
                        else 0,
                    )
                    print("[menu] Waterfall watcher launched.")
                    
            elif cmd == "quit":
                print("[menu] Quit received — stopping observation.")
                # Clean up any launched subprocesses
                for name, proc in watchers.items():
                    if proc.poll() is None:
                        print(f"[menu] Terminating {name} watcher.")
                        proc.terminate()
                stop_event.set()
                break

            else:
                print(f"[menu] Unknown command: '{cmd}'")
                
        except EOFError:
            break

    print("[menu] Command menu exiting.")

# ---------------------------------------------------------------------------
# Main observing loop
# ---------------------------------------------------------------------------

def main():
    print("=" * 60)
    print("X-Band Interferometer — Sun Test Observation")
    print("=" * 60)

    # --- Initialize hardware ---
    print("[init] Connecting to interferometer...")
    ifm = interf.Interferometer()

    print("[init] Connecting to SNAP correlator...")
    spec = snap.UGRadioSnap()
    spec.initialize(mode='corr, force=True)
    print("[init] SNAP initialized.")

    # --- Check Sun is up before starting ---
    result = point_to_sun(ifm)
    if result is None:
        print("[main] Sun is below minimum elevation. Exiting.")
        ifm.stow()
        return

    # --- Start collector thread ---
    collector = threading.Thread(target=collect_data, args=(spec,), daemon=True)
    collector.start()

     # --- Start command menu thread ---
    menu = threading.Thread(target=command_menu, daemon=True)
    menu.start()
    
    # --- Main loop: update pointing every POINT_UPDATE_SEC ---
    t_start = time.time()
    last_save_time = t_start

    try:
        while not stop_event.is_set():
            elapsed = time.time() - t_start

            # Update pointing
            result = point_to_sun(ifm)
            if result is None:
                print("[main] Sun below minimum altitude — holding position, continuing observation.")
            else:
                print(f"[pointing] Pointing updated: alt={result[0]:.1f}°, az={result[1]:.1f}°")

            # Periodic mid-observation save (every 5 minutes) as a safety backup
            if time.time() - last_save_time > 300:
                save_data(tag="_partial")
                last_save_time = time.time()

            remaining_buf = len(data_buffer['times'])
            print(f"[main] {elapsed/60:.1f} min elapsed. "
                  f"Buffer size: {remaining_buf} integrations.")

            time.sleep(POINT_UPDATE_SEC)

    except KeyboardInterrupt:
        print("\n[main] Interrupted by user.")

    finally:
        # --- Shutdown ---
        print("[main] Stopping collector thread...")
        stop_event.set()
        collector.join(timeout=5)
        menu.join(timeout=2)

        # --- Save final data ---
        fname = save_data(tag="_final")

        # --- Stow telescope ---
        print("[main] Stowing telescope.")
        ifm.stow()

        print("[main] Done.")
        if fname:
            print(f"[main] Data saved to: {fname}")

# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()
