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

MAX_SAMPLES = 5000
SAVE_INTERVAL = 600   # seconds (auto-save every 10 min)

OUTPUT_DIR = "data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# Global buffers
# -----------------------------
phase_series = []
A_matrix = []
time_series = []

vis_buffer = []
jd_buffer = []
alt_buffer = []
az_buffer = []

waterfall_amp = []
waterfall_phase = []

# -----------------------------
# Time tracking
# -----------------------------
t_start_unix = time.time()
jd_start = ugradio.timing.julian_date(t_start_unix)

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
# Data + visualization loop
# -----------------------------
def data_loop():
    print("[DATA] Thread started")

    while True:
        try:
            jd = current_jd()

            ra, dec = ugradio.coord.sunpos(jd)
            alt, az = ugradio.coord.get_altaz(ra, dec, jd, LAT, LON, ALT)

            vis = spec.read_data()

            amp = np.abs(vis)
            phase = np.angle(vis)

            # Store globally (already in your script)
            vis_buffer.append(vis)
            jd_buffer.append(jd)
            alt_buffer.append(alt)
            az_buffer.append(az)

            waterfall_amp.append(amp)
            waterfall_phase.append(phase)

            if len(waterfall_amp) > 200:
                waterfall_amp.pop(0)
                waterfall_phase.pop(0)

            # Phase tracking
            phi = phase[CHANNEL]
            s = radec_to_unit(ra, dec)
            A_row = (2 * np.pi / lam) * s

            phase_series.append(phi)
            A_matrix.append(A_row)
            
        if time.time() - t_start_unix > SAVE_INTERVAL:
            save_data()

        time.sleep(DATA_INTERVAL)

         except Exception as e:
            print("[DATA ERROR]", e)
            time.sleep(1)

def plotting_loop():
    plt.ion()
    fig, axs = plt.subplots(4, 1, figsize=(10, 12))

    while True:
        try:
            if len(waterfall_amp) == 0:
                time.sleep(1)
                continue

            axs[0].cla()
            axs[1].cla()
            axs[2].cla()
            axs[3].cla()

            # Latest data
            amp = waterfall_amp[-1]
            phase = waterfall_phase[-1]

            axs[0].plot(amp)
            axs[0].set_title("Amplitude")

            axs[1].plot(phase)
            axs[1].set_title("Phase")

            axs[2].imshow(np.array(waterfall_amp), aspect='auto', origin='lower')
            axs[2].set_title("Amplitude Waterfall")

            if len(phase_series) > 10:
                phi_array = np.unwrap(np.array(phase_series))
                axs[3].plot(phi_array)
                axs[3].set_title("Fringe Phase")

            plt.pause(0.01)

        except Exception as e:
            print("[PLOT ERROR]", e)
            time.sleep(1)

# -----------------------------
# Save function
# -----------------------------
def save_data():
    global t_start_unix, jd_start
    global vis_buffer, jd_buffer, alt_buffer, az_buffer

    if len(vis_buffer) == 0:
        print("No data to save.")
        return

    t_end_unix = time.time()
    jd_end = current_jd()

    timestamp = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"solar_obs_{timestamp}.npz")

    print(f"\nSaving → {filename}")

    np.savez(
        filename,

        # Time
        t_start_unix=t_start_unix,
        t_end_unix=t_end_unix,
        jd_start=jd_start,
        jd_end=jd_end,

        # Location
        lat=LAT,
        lon=LON,
        alt_site=ALT,

        # Instrument
        frequency_hz=f,
        wavelength_m=lam,

        # Pointing
        alt_array=np.array(alt_buffer),
        az_array=np.array(az_buffer),

        # Data
        visibilities=np.array(vis_buffer),
        jd_array=np.array(jd_buffer),

        # Fringe solving
        phase_series=np.array(phase_series),
        A_matrix=np.array(A_matrix)
    )

    print("Saved successfully.")

    # Reset buffers for next block
    vis_buffer.clear()
    jd_buffer.clear()
    alt_buffer.clear()
    az_buffer.clear()

    t_start_unix = time.time()
    jd_start = current_jd()

# -----------------------------
# Baseline solver
# -----------------------------
def solve_baseline():
    if len(phase_series) < 20:
        print("Not enough data yet.")
        return

    phi_array = np.unwrap(np.array(phase_series))
    A = np.array(A_matrix)

    B_fit, _, _, _ = np.linalg.lstsq(A, phi_array, rcond=None)

    print("\n===== BASELINE ESTIMATE =====")
    print(f"Bx (E-W): {B_fit[0]:.3f} m")
    print(f"By (N-S): {B_fit[1]:.3f} m")
    print(f"Bz (Up):  {B_fit[2]:.3f} m")
    print(f"Length:   {np.linalg.norm(B_fit):.3f} m")

    phi_model = A @ B_fit

    plt.figure()
    plt.plot(phi_array, label="Measured")
    plt.plot(phi_model, label="Model")
    plt.legend()
    plt.title("Phase Fit Comparison")
    plt.show()

# -----------------------------
# Main
# -----------------------------
try:
    print("Starting system...")

    t1 = threading.Thread(target=tracking_loop, daemon=True)
    t2 = threading.Thread(target=data_loop, daemon=True)

    t1.start()
    t2.start()

    plotting_loop()
    
    while True:
        cmd = input("\nCommands: solve / save / quit : ").strip().lower()

        if cmd == "solve":
            solve_baseline()

        elif cmd == "save":
            save_data()

        elif cmd == "quit":
            break

except KeyboardInterrupt:
    print("\nStopping system...")

finally:
    ifm.stow()
