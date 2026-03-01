import numpy as np
import ugradio
import time

# ---------- SDR SETTINGS ----------
CENTER_FREQ = 1420e6
SAMPLE_RATE = 100000  # very safe for Pi
GAIN = 20
CHUNK_SIZE = 2048     # small chunk for USB
SLEEP_TIME = 0.005
BLOCK_TIME = 0.05     # 50 ms per burst
NUM_BLOCKS = 10       # average over 10 bursts
# -----------------------------------

def snapshot_capture():
    """Capture a single short burst (50ms) and stop SDR."""
    sdr = ugradio.sdr.SDR(center_freq=CENTER_FREQ,
                          sample_rate=SAMPLE_RATE,
                          gain=GAIN)
    time.sleep(0.05)  # allow SDR to stabilize

    collected = 0
    data_list = []
    total_samples = int(SAMPLE_RATE * BLOCK_TIME)

    while collected < total_samples:
        try:
            d = sdr.capture_data(CHUNK_SIZE)
            data_list.append(d)
            collected += len(d)
            time.sleep(SLEEP_TIME)
        except Exception as e:
            print("USB retry:", e)

    sdr.stop()  # stop streaming and release USB
    return np.concatenate(data_list)

input("next part")

input_power_dbm = -40
R = 50

V_rms_true = np.sqrt(10**((input_power_dbm - 30)/10) * R)
V_meas_list = []

print("Capturing calibration bursts...")
for _ in range(NUM_BLOCKS):
    data = snapshot_capture()
    I = np.real(data)
    Q = np.imag(data)
    V_meas_list.append(np.sqrt(np.mean(I**2 + Q**2)))

V_meas = np.mean(V_meas_list)
K = V_rms_true / V_meas

print("True Vrms:", V_rms_true)
print("Average digital RMS:", V_meas)
print("Voltage scale factor (V/unit):", K)

input("next part")

c = 3e8
L = 5.0

def measure_phase_snapshot():
    phases = []
    for _ in range(NUM_BLOCKS):
        data = snapshot_capture()
        phasor = np.mean(data)
        phases.append(np.angle(phasor))
    return np.mean(phases)

print("Measure SHORT cable...")
phi_short = measure_phase_snapshot()

print("Measure LONG cable...")
phi_long = measure_phase_snapshot()

delta_phi = np.angle(np.exp(1j*(phi_long - phi_short)))
v = (2 * np.pi * CENTER_FREQ * L) / delta_phi
VF = v / c

print("Phase difference (rad):", delta_phi)
print("Cable velocity (m/s):", v)
print("Velocity factor:", VF)

input("next part")

L = 5.0

def measure_power_snapshot():
    powers = []
    for _ in range(NUM_BLOCKS):
        data = snapshot_capture()
        powers.append(10 * np.log10(np.mean(np.abs(data)**2)))
    return np.mean(powers)

print("Measure SHORT cable power...")
P_short = measure_power_snapshot()

print("Measure LONG cable power...")
P_long = measure_power_snapshot()

delta_P = P_short - P_long
alpha_db_per_m = delta_P / L

print("Short power (dB):", P_short)
print("Long power (dB):", P_long)
print("Total loss (dB):", delta_P)
print("Loss per meter (dB/m):", alpha_db_per_m)
