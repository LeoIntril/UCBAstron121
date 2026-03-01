import numpy as np
import ugradio
import time

# ---------- SAFE SDR SETTINGS ----------
CENTER_FREQ = 1420e6
SAMPLE_RATE = 100000      # 100 kS/s (very stable)
GAIN = 20
CHUNK = 8192              # Safe USB block size
SLEEP = 0.005             # Allow USB drain
# ---------------------------------------

def safe_capture(integration_time):
    sdr = ugradio.sdr.SDR(lo_freq=CENTER_FREQ,
                          samp_rate=SAMPLE_RATE,
                          gain=GAIN)
    
    total_samples = int(SAMPLE_RATE * integration_time)
    collected = 0
    data = []

    while collected < total_samples:
        try:
            d = sdr.capture_data(CHUNK)
            data.append(d)
            collected += len(d)
            time.sleep(SLEEP)
        except Exception as e:
            print("USB retry:", e)

    return np.concatenate(data)

# ---------- USER INPUT ----------
input_power_dbm = -40
R = 50
integration_time = 1
# --------------------------------

# Convert dBm → true Vrms
P_watts = 10**((input_power_dbm - 30)/10)
V_rms_true = np.sqrt(P_watts * R)

print("Capturing calibration tone...")
data = safe_capture(integration_time)

I = np.real(data)
Q = np.imag(data)

V_meas = np.sqrt(np.mean(I**2 + Q**2))

K = V_rms_true / V_meas

print("True Vrms:", V_rms_true)
print("Measured digital RMS:", V_meas)
print("Calibration factor (Volts per digital unit):", K)

V_actual = K * np.sqrt(I**2 + Q**2)

input("Set up next part and press enter")

c = 3e8
L = 5.0  # length of long cable (meters)

def measure_phase():
    data = safe_capture(1)
    phasor = np.mean(data)
    return np.angle(phasor)

print("Measure SHORT cable...")
phi_short = measure_phase()

print("Measure LONG cable...")
phi_long = measure_phase()

# Unwrap phase properly
delta_phi = np.angle(np.exp(1j*(phi_long - phi_short)))

v = (2 * np.pi * CENTER_FREQ * L) / delta_phi
VF = v / c

print("Phase difference (rad):", delta_phi)
print("Velocity (m/s):", v)
print("Velocity factor:", VF)

input("Set up next part and press enter")

L = 5.0  # cable length in meters

def measure_power_db():
    data = safe_capture(1)
    power_linear = np.mean(np.abs(data)**2)
    return 10 * np.log10(power_linear)

print("Measure SHORT cable power...")
P_short = measure_power_db()

print("Measure LONG cable power...")
P_long = measure_power_db()

delta_P = P_short - P_long
alpha_db_per_m = delta_P / L

print("Short power (dB):", P_short)
print("Long power (dB):", P_long)
print("Total loss (dB):", delta_P)
print("Loss per meter (dB/m):", alpha_db_per_m)
