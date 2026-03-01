import numpy as np
import ugradio
import time
import os

# ---------- SDR SETTINGS ----------
CENTER_FREQ = 1420e6   # Hydrogen line
SAMPLE_RATE = 250000   # reliable low rate for Pi + NESDR SMArt
GAIN = 20
CHUNK_SIZE = 2048
BURST_TIME = 0.05      # 50 ms per burst
NUM_BURSTS = 10
SLEEP_TIME = 0.005
# -----------------------------------

def snapshot_capture():
    """
    Capture a short burst (~50 ms) of IQ samples and stop SDR.
    """
    sdr = ugradio.sdr.SDR(
        lo_freq=CENTER_FREQ,
        samp_rate=SAMPLE_RATE,
        gain=GAIN,
        direct=False    # IMPORTANT: disable direct sampling
    )
    time.sleep(0.05)  # allow tuner to settle

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

def save_iq_data(filename, iq_data, metadata=None):
    """
    Save IQ samples and optional metadata to a .npz file
    """
    if metadata is None:
        metadata = {}
    folder = os.path.dirname(filename)
    if folder and not os.path.exists(folder):
        os.makedirs(folder)
    np.savez(filename, iq=iq_data, **metadata)
    print(f"Saved data to {filename}")

input("next part")

input_power_dbm = -40  # generator output
R = 50                  # system impedance

V_rms_true = np.sqrt(10**((input_power_dbm - 30)/10) * R)
V_meas_list = []

print("Capturing calibration bursts...")
all_bursts = []
for _ in range(NUM_BURSTS):
    data = snapshot_capture()
    I = np.real(data)
    Q = np.imag(data)
    V_meas_list.append(np.sqrt(np.mean(I**2 + Q**2)))
    all_bursts.append(data)

V_meas = np.mean(V_meas_list)
K = V_rms_true / V_meas

print("True Vrms:", V_rms_true)
print("Average digital RMS:", V_meas)
print("Voltage scale factor (V/unit):", K)

# Save IQ bursts and metadata
metadata = {
    'input_power_dbm': input_power_dbm,
    'R': R,
    'sample_rate': SAMPLE_RATE,
    'num_bursts': NUM_BURSTS,
    'burst_time_s': BURST_TIME,
    'gain': GAIN
}
save_iq_data("part1_calibration.npz", np.array(all_bursts), metadata)

input("next part")

c = 3e8
L = 5.0  # cable length in meters

def measure_phase_bursts():
    phases = []
    all_bursts = []
    for _ in range(NUM_BURSTS):
        data = snapshot_capture()
        phasor = np.mean(data)
        phases.append(np.angle(phasor))
        all_bursts.append(data)
    return np.mean(phases), np.array(all_bursts)

print("Measure SHORT cable...")
phi_short, short_bursts = measure_phase_bursts()

print("Measure LONG cable...")
phi_long, long_bursts = measure_phase_bursts()

delta_phi = np.angle(np.exp(1j*(phi_long - phi_short)))
v = (2 * np.pi * CENTER_FREQ * L) / delta_phi
VF = v / c

print("Phase difference (rad):", delta_phi)
print("Cable velocity (m/s):", v)
print("Velocity factor:", VF)

# Save IQ bursts and metadata
metadata = {
    'sample_rate': SAMPLE_RATE,
    'num_bursts': NUM_BURSTS,
    'burst_time_s': BURST_TIME,
    'cable_length_m': L,
    'gain': GAIN
}
save_iq_data("part2_velocity.npz", np.array([short_bursts, long_bursts]), metadata)

input("next part")

c = 3e8
L = 5.0  # cable length in meters

def measure_phase_bursts():
    phases = []
    all_bursts = []
    for _ in range(NUM_BURSTS):
        data = snapshot_capture()
        phasor = np.mean(data)
        phases.append(np.angle(phasor))
        all_bursts.append(data)
    return np.mean(phases), np.array(all_bursts)

print("Measure SHORT cable...")
phi_short, short_bursts = measure_phase_bursts()

print("Measure LONG cable...")
phi_long, long_bursts = measure_phase_bursts()

delta_phi = np.angle(np.exp(1j*(phi_long - phi_short)))
v = (2 * np.pi * CENTER_FREQ * L) / delta_phi
VF = v / c

print("Phase difference (rad):", delta_phi)
print("Cable velocity (m/s):", v)
print("Velocity factor:", VF)

# Save IQ bursts and metadata
metadata = {
    'sample_rate': SAMPLE_RATE,
    'num_bursts': NUM_BURSTS,
    'burst_time_s': BURST_TIME,
    'cable_length_m': L,
    'gain': GAIN
}
save_iq_data("part2_velocity.npz", np.array([short_bursts, long_bursts]), metadata)

save_iq_data("part3_loss.npz", all_powers, metadata)
