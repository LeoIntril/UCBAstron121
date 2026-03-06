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
all_bursts = []

print("Capturing calibration bursts...")

for i in range(NUM_BURSTS):
    data = snapshot_capture()

    I = np.real(data)
    Q = np.imag(data)

    Vrms = np.sqrt(np.mean(I**2 + Q**2))
    V_meas_list.append(Vrms)

    all_bursts.append(data)

V_meas_array = np.array(V_meas_list)

# Mean and error
V_meas_mean = np.mean(V_meas_array)
V_meas_std = np.std(V_meas_array, ddof=1)
V_meas_err = V_meas_std / np.sqrt(NUM_BURSTS)

# Voltage scale factor
K = V_rms_true / V_meas_mean

print("\nCalibration results")
print("True Vrms:", V_rms_true)
print("Measured Vrms mean:", V_meas_mean)
print("Measured Vrms std:", V_meas_std)
print("Measured Vrms standard error:", V_meas_err)
print("Voltage scale factor (V/unit):", K)

# Save IQ bursts and metadata
metadata = {
    'input_power_dbm': input_power_dbm,
    'R': R,
    'sample_rate': SAMPLE_RATE,
    'num_bursts': NUM_BURSTS,
    'burst_time_s': BURST_TIME,
    'gain': GAIN,
    'Vrms_mean': V_meas_mean,
    'Vrms_std': V_meas_std,
    'Vrms_err': V_meas_err,
    'scale_factor_K': K
}

save_iq_data("part1_calibration.npz", np.array(all_bursts), metadata)

input("next part")
import numpy as np
import ugradio
import time
import os

# ---------- SDR SETTINGS ----------
CENTER_FREQ = 1420e6
SAMPLE_RATE = 250000
GAIN = 20
CHUNK_SIZE = 2048
BURST_TIME = 0.05
NUM_TRIALS = 10
SLEEP_TIME = 0.005
# -----------------------------------

# Known cable lengths
L_short = 0.2   # meters
L_long  = 5.0   # meters

# -----------------------------------
# SDR Snapshot Capture
# -----------------------------------

def snapshot_capture():
    sdr = ugradio.sdr.SDR(
        lo_freq=CENTER_FREQ,
        samp_rate=SAMPLE_RATE,
        gain=GAIN,
        direct=False
    )

    time.sleep(0.05)

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

    sdr.stop()

    return np.concatenate(data_list)

# -----------------------------------
# Voltage Measurement
# -----------------------------------

def measure_voltage_trials(label):

    voltages = []
    bursts = []

    print(f"\nMeasuring {label} cable...")

    for i in range(NUM_TRIALS):

        data = snapshot_capture()

        I = np.real(data)
        Q = np.imag(data)

        Vrms = np.sqrt(np.mean(I**2 + Q**2))

        voltages.append(Vrms)
        bursts.append(data)

        print(f"Trial {i+1}: Vrms = {Vrms}")

    voltages = np.array(voltages)

    mean = np.mean(voltages)
    std = np.std(voltages, ddof=1)
    err = std / np.sqrt(NUM_TRIALS)

    return mean, std, err, voltages, np.array(bursts)

# -----------------------------------
# Measurements
# -----------------------------------

input("Connect SHORT cable and press Enter")

V_short_mean, V_short_std, V_short_err, V_short_trials, short_bursts = measure_voltage_trials("SHORT")

input("\nConnect LONG cable and press Enter")

V_long_mean, V_long_std, V_long_err, V_long_trials, long_bursts = measure_voltage_trials("LONG")

# -----------------------------------
# Attenuation Calculations
# -----------------------------------

# Voltage ratio
ratio = V_long_mean / V_short_mean

# Propagate error
ratio_err = ratio * np.sqrt(
    (V_long_err / V_long_mean)**2 +
    (V_short_err / V_short_mean)**2
)

# Power loss in dB
atten_dB = -20 * np.log10(ratio)

atten_err = (20 / np.log(10)) * (ratio_err / ratio)

# Cable length difference
L_diff = L_long - L_short

# Loss per meter
loss_per_m = atten_dB / L_diff
loss_per_m_err = atten_err / L_diff

# -----------------------------------
# Results
# -----------------------------------

print("\n========== RESULTS ==========")

print("Short cable Vrms:", V_short_mean, "+/-", V_short_err)
print("Long cable Vrms:", V_long_mean, "+/-", V_long_err)

print("\nVoltage ratio (long/short):", ratio, "+/-", ratio_err)

print("\nTotal attenuation (dB):", atten_dB, "+/-", atten_err)

print("\nCable attenuation (dB/m):", loss_per_m, "+/-", loss_per_m_err)

print("=============================")

# -----------------------------------
# Save Data
# -----------------------------------

metadata = {
    "short_length_m": L_short,
    "long_length_m": L_long,
    "num_trials": NUM_TRIALS,
    "sample_rate": SAMPLE_RATE,
    "gain": GAIN,

    "V_short_mean": V_short_mean,
    "V_short_err": V_short_err,

    "V_long_mean": V_long_mean,
    "V_long_err": V_long_err,

    "atten_dB": atten_dB,
    "atten_err": atten_err,

    "loss_per_meter": loss_per_m,
    "loss_per_meter_err": loss_per_m_err
}

np.savez(
    "cable_loss_measurement.npz",
    short_trials=V_short_trials,
    long_trials=V_long_trials,
    short_bursts=short_bursts,
    long_bursts=long_bursts,
    **metadata
)

print("\nData saved to cable_loss_measurement.npz")
