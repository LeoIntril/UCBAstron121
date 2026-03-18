import ugradio
import snap_spec.snap
import numpy as np
import time
import threading
import datetime
import os

# -----------------------------
# Configuration
# -----------------------------
LAT = 37.87        # Berkeley latitude (deg)
LON = -122.26      # Berkeley longitude (deg)
ALT = 0.0          # altitude (meters, adjust if needed)

POINTING_INTERVAL = 30   # seconds between repointing
DATA_INTERVAL = 5        # seconds between data reads
OUTPUT_DIR = "data"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Initialize hardware
# -----------------------------
ifm = ugradio.interf.Interferometer()
spec = snap_spec.snap.UGRadioSnap()

print("Initializing SNAP correlator...")
spec.initialize(mode='corr')

# -----------------------------
# Helper: get current JD
# -----------------------------
def current_jd():
    return ugradio.timing.julian_date(time.time())

# -----------------------------
# Helper: compute Sun Alt/Az
# -----------------------------
def get_sun_altaz(jd):
    ra, dec = ugradio.coord.sunpos(jd)
    
    # Precess coordinates to current epoch
    ra, dec = ugradio.coord.precess(ra, dec, jd, 2000)
    
    alt, az = ugradio.coord.get_altaz(ra, dec, jd, LAT, LON, ALT)
    return alt, az, ra, dec

# -----------------------------
# Telescope tracking loop
# -----------------------------
def tracking_loop():
    while True:
        jd = current_jd()
        alt, az, ra, dec = get_sun_altaz(jd)

        print(f"[TRACK] JD={jd:.5f} ALT={alt:.2f} AZ={az:.2f}")
        
        # Only point if above horizon
        if alt > 0:
            ifm.point(alt, az)
        else:
            print("[TRACK] Sun below horizon, stowing.")
            ifm.stow()

        time.sleep(POINTING_INTERVAL)

# -----------------------------
# Data acquisition loop
# -----------------------------
def data_loop():
    while True:
        jd = current_jd()
        alt, az, ra, dec = get_sun_altaz(jd)

        # Read correlator data
        spectrum = spec.read_data()

        timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(OUTPUT_DIR, f"spectrum_{timestamp}.npz")

        np.savez(
            filename,
            jd=jd,
            alt=alt,
            az=az,
            ra=ra,
            dec=dec,
            spectrum=spectrum
        )

        print(f"[DATA] Saved {filename}")

        time.sleep(DATA_INTERVAL)

# -----------------------------
# Start threads
# -----------------------------
try:
    print("Starting tracking and data acquisition...")
    
    t1 = threading.Thread(target=tracking_loop, daemon=True)
    t2 = threading.Thread(target=data_loop, daemon=True)

    t1.start()
    t2.start()

    # Keep main thread alive
    while True:
        time.sleep(1)

except KeyboardInterrupt:
    print("\nShutting down...")
    ifm.stow()
