import ugradio
import snap_spec.snap
import numpy as np
import time
import threading
import matplotlib.pyplot as plt
import datetime
import os

# -----------------------------
# Configuration
# -----------------------------
LAT = ugradio.nch.lat
LON = ugradio.nch.lon
ALT = 120

c = 3e8
f = 10.5e9
lam = c / f

POINTING_INTERVAL = 10     # seconds
DATA_INTERVAL = 2          # seconds
CHANNEL = 100              # frequency bin to track

MAX_SAMPLES = 2000         # limit memory

# -----------------------------
# Storage
# -----------------------------
phase_series = []
A_matrix = []
time_series = []

waterfall_amp = []
waterfall_phase = []

# -----------------------------
# Initialize hardware
# -----------------------------
ifm = ugradio.interf.Interferometer()
spec = snap_spec.snap.UGRadioSnap()

print("Initializing SNAP...")
spec.initialize(mode='corr')

# -----------------------------
def current_jd():
    return ugradio.timing.julian_date(time.time())

# -----------------------------
def radec_to_unit(ra, dec):
    return np.array([
        np.cos(dec) * np.cos(ra),
        np.cos(dec) * np.sin(ra),
        np.sin(dec)
    ])

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
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))

    while True:
        jd = current_jd()

        # Sun position
        ra, dec = ugradio.coord.sunpos(jd)
        ra, dec = ugradio.coord.precess(ra, dec, jd, 2000)

        s = radec_to_unit(ra, dec)

        # Read correlator
        vis = spec.read_data()

        # Extract amplitude + phase
        amp = np.abs(vis)
        phase = np.angle(vis)

        # Store waterfall data
        waterfall_amp.append(amp)
        waterfall_phase.append(phase)

        if len(waterfall_amp) > 200:
            waterfall_amp.pop(0)
            waterfall_phase.pop(0)

        # -----------------------------
        # Phase tracking for baseline solve
        # -----------------------------
        phi = phase[CHANNEL]

        A_row = (2 * np.pi / lam) * s

        phase_series.append(phi)
        A_matrix.append(A_row)
        time_series.append(jd)

        if len(phase_series) > MAX_SAMPLES:
            phase_series.pop(0)
            A_matrix.pop(0)
            time_series.pop(0)

        # -----------------------------
        # Plotting
        # -----------------------------
        axs[0].cla()
        axs[1].cla()
        axs[2].cla()
        axs[3].cla()

        # Raw amplitude
        axs[0].plot(amp)
        axs[0].set_title("Visibility Amplitude")

        # Raw phase
        axs[1].plot(phase)
        axs[1].set_title("Visibility Phase")

        # Waterfall amplitude
        axs[2].imshow(
            np.array(waterfall_amp),
            aspect='auto',
            origin='lower'
        )
        axs[2].set_title("Amplitude Waterfall")

        # Phase vs time (unwrapped)
        if len(phase_series) > 10:
            phi_array = np.unwrap(np.array(phase_series))
            axs[3].plot(phi_array)
            axs[3].set_title("Fringe Phase vs Time (Unwrapped)")

        plt.pause(0.01)

        time.sleep(DATA_INTERVAL)

# -----------------------------
# Baseline solver (run anytime)
# -----------------------------
def solve_baseline():
    if len(phase_series) < 20:
        print("Not enough data yet.")
        return

    phi_array = np.unwrap(np.array(phase_series))
    A = np.array(A_matrix)

    B_fit, _, _, _ = np.linalg.lstsq(A, phi_array, rcond=None)

    print("\n===== BASELINE ESTIMATE =====")
    print("Bx (E-W): {:.3f} m".format(B_fit[0]))
    print("By (N-S): {:.3f} m".format(B_fit[1]))
    print("Bz (Up):  {:.3f} m".format(B_fit[2]))
    print("Length:   {:.3f} m".format(np.linalg.norm(B_fit)))

    # Compare model vs data
    phi_model = A @ B_fit

    plt.figure()
    plt.plot(phi_array, label="Measured")
    plt.plot(phi_model, label="Model")
    plt.legend()
    plt.title("Phase Fit Comparison")
    plt.show()

# -----------------------------
# Run threads
# -----------------------------
try:
    print("Starting system...")

    t1 = threading.Thread(target=tracking_loop, daemon=True)
    t2 = threading.Thread(target=data_loop, daemon=True)

    t1.start()
    t2.start()

    while True:
        cmd = input("\nType 'solve' to estimate baseline: ")

        if cmd.strip().lower() == "solve":
            solve_baseline()

except KeyboardInterrupt:
    print("\nStopping system...")
    ifm.stow()

import datetime
import os

def save_data():
    if len(vis_buffer) == 0:
        print("No data to save.")
        return

    t_end_unix = time.time()
    jd_end = current_jd()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = f"solar_obs_{timestamp}.npz"

    print(f"\nSaving data to {filename}...")

    np.savez(
        filename,

        # -----------------
        # Time metadata
        # -----------------
        t_start_unix=t_start_unix,
        t_end_unix=t_end_unix,
        jd_start=jd_start,
        jd_end=jd_end,

        # -----------------
        # Location
        # -----------------
        lat=LAT,
        lon=LON,
        alt=ALT,

        # -----------------
        # Instrument
        # -----------------
        frequency_hz=f,
        wavelength_m=lam,

        # -----------------
        # Data
        # -----------------
        visibilities=np.array(vis_buffer),
        jd_array=np.array(jd_buffer),
        phase_series=np.array(phase_series),
        A_matrix=np.array(A_matrix)
    )

    print("Save complete.")
