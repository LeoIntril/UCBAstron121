import ugradio
import snap_spec.snap
import numpy as np
import time
import threading
import matplotlib.pyplot as plt

# -----------------------------
# Constants
# -----------------------------
LAT = 37.87
LON = -122.26
ALT = 0.0

c = 3e8
f = 10.5e9
lam = c / f

POINTING_INTERVAL = 10
DATA_INTERVAL = 2

# Example baseline (meters) — UPDATE THIS
baseline = np.array([5.0, 0.0, 0.0])  # East-West 5 m

# Storage for waterfall
waterfall = []

# -----------------------------
# Initialize hardware
# -----------------------------
ifm = ugradio.interf.Interferometer()
spec = snap_spec.snap.UGRadioSnap()
spec.initialize(mode='corr')

# -----------------------------
def current_jd():
    return ugradio.timing.julian_date(time.time())

# -----------------------------
# Convert RA/Dec to unit vector
# -----------------------------
def radec_to_unit(ra, dec):
    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ])

# -----------------------------
# Fringe phase calculation
# -----------------------------
def fringe_phase(baseline, source_vec):
    return (2 * np.pi / lam) * np.dot(baseline, source_vec)

# -----------------------------
# Tracking loop
# -----------------------------
def tracking_loop():
    while True:
        jd = current_jd()
        ra, dec = ugradio.coord.sunpos(jd)
        ra, dec = ugradio.coord.precess(ra, dec, jd, 2000)

        alt, az = ugradio.coord.get_altaz(ra, dec, jd, LAT, LON, ALT)

        if alt > 0:
            ifm.point(alt, az)
        else:
            ifm.stow()

        time.sleep(POINTING_INTERVAL)

# -----------------------------
# Data + Visualization loop
# -----------------------------
def data_loop():
    plt.ion()
    fig, axs = plt.subplots(3, 1, figsize=(8, 10))

    while True:
        jd = current_jd()
        ra, dec = ugradio.coord.sunpos(jd)
        ra, dec = ugradio.coord.precess(ra, dec, jd, 2000)

        source_vec = radec_to_unit(ra, dec)

        # Read correlator output
        vis = spec.read_data()

        # Compute fringe phase
        phi = fringe_phase(baseline, source_vec)

        # Apply fringe stopping
        vis_fs = vis * np.exp(-1j * phi)

        # Store for waterfall
        waterfall.append(np.abs(vis_fs))

        if len(waterfall) > 100:
            waterfall.pop(0)

        # -----------------------------
        # Plotting
        # -----------------------------
        axs[0].cla()
        axs[1].cla()
        axs[2].cla()

        axs[0].plot(np.abs(vis))
        axs[0].set_title("Raw Visibility Amplitude")

        axs[1].plot(np.abs(vis_fs))
        axs[1].set_title("Fringe-Stopped Visibility")

        axs[2].imshow(
            np.array(waterfall),
            aspect='auto',
            origin='lower'
        )
        axs[2].set_title("Waterfall (Fringe-Stopped)")

        plt.pause(0.01)

        time.sleep(DATA_INTERVAL)

# -----------------------------
# Run threads
# -----------------------------
try:
    t1 = threading.Thread(target=tracking_loop, daemon=True)
    t2 = threading.Thread(target=data_loop, daemon=True)

    t1.start()
    t2.start()

    while True:
        time.sleep(1)

except KeyboardInterrupt:
    ifm.stow()
    print("Stopped.")
