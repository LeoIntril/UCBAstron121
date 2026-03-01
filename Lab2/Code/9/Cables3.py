import numpy as np
import ugradio
import time

# ---------- SDR SETTINGS ----------
CENTER_FREQ = 1420e6
SAMPLE_RATE = 1e6     # 1 MS/s, works reliably
GAIN = 20
CHUNK_SIZE = 2048     # small chunk for Pi USB
BURST_TIME = 0.05     # 50 ms per burst
NUM_BURSTS = 10       # average over 10 bursts
SLEEP_TIME = 0.005    # allow USB buffer drain
# -----------------------------------

def snapshot_capture():
    """
    Capture a short burst (~50 ms) of IQ samples and stop SDR.
    Returns concatenated IQ array.
    """
    sdr = ugradio.sdr.SDR(
        center_freq=CENTER_FREQ,
        sample_rate=SAMPLE_RATE,
        gain=GAIN,
        direct=False       # disable direct sampling!
    )
    time.sleep(0.05)  # allow tuner to stabilize

    total_samples = int(SAMPLE_RATE * BURST_TIME)
    collected = 0
    data_list = []

    while collected < total_samples:
        try:
            d = sdr.capture_data(CHUNK_SIZE)
            data_list.append(d)
            collected += len(d)
            time.sleep(SLEEP_TIME)
        except Exception as e:
            print("USB retry:", e)

    sdr.stop()  # release USB streaming
    return np.concatenate(data_list)

input("next part")

input_power_dbm = -40  # generator output
R = 50                  # system impedance in ohms

V_rms_true = np.sqrt(10**((input_power_dbm - 30)/10) * R)
V_meas_list = []

print("Capturing calibration bursts...")
for _ in range(NUM_BURSTS):
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
L = 5.0  # cable length in meters

def measure_phase_snapshot():
    phases = []
    for _ in range(NUM_BURSTS):
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

L = 5.0  # cable length in meters

def measure_power_snapshot():
    powers = []
    for _ in range(NUM_BURSTS):
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

input("Save Data")

import os

def save_iq_data(filename, iq_data, metadata=None):
    """
    Saves IQ samples and optional metadata to a .npz file.
    """
    if metadata is None:
        metadata = {}
    
    # Ensure folder exists
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    
    np.savez(filename, iq=iq_data, **metadata)
    print(f"Saved data to {filename}")

# Save raw bursts and calibration metadata
metadata = {
    'input_power_dbm': input_power_dbm,
    'R': R,
    'sample_rate': SAMPLE_RATE,
    'num_bursts': NUM_BURSTS,
    'burst_time_s': BURST_TIME,
    'gain': GAIN
}

# Combine all bursts into a single array
all_bursts = np.array(V_meas_list)  # digital RMS per burst
save_iq_data("part1_calibration.npz", all_bursts, metadata)

# Save the phase measurements per burst
metadata = {
    'sample_rate': SAMPLE_RATE,
    'num_bursts': NUM_BURSTS,
    'burst_time_s': BURST_TIME,
    'cable_length_m': L,
    'gain': GAIN
}

all_phases = np.array([np.angle(np.mean(snapshot_capture())) for _ in range(NUM_BURSTS)])
save_iq_data("part2_velocity.npz", all_phases, metadata)

# Save the power measurements per burst
metadata = {
    'sample_rate': SAMPLE_RATE,
    'num_bursts': NUM_BURSTS,
    'burst_time_s': BURST_TIME,
    'cable_length_m': L,
    'gain': GAIN
}

all_powers = np.array([10*np.log10(np.mean(np.abs(snapshot_capture())**2)) for _ in range(NUM_BURSTS)])
save_iq_data("part3_loss.npz", all_powers, metadata)
