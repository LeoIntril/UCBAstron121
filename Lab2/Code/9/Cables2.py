import numpy as np
import ugradio
import time

# ---------- SDR SETTINGS ----------
CENTER_FREQ = 1420e6
SAMPLE_RATE = 250000   # valid for NESDR SMArt
GAIN = 20
CHUNK_SIZE = 8192      # small chunk to prevent USB overflow
SLEEP_TIME = 0.005     # allow USB buffer to drain
# -----------------------------------

def safe_capture(integration_time):
    """
    Captures IQ samples safely from the SDR in small chunks.
    """
    sdr = ugradio.sdr.SDR(lo_freq=CENTER_FREQ,
                          samp_rate=SAMPLE_RATE,
                          gain=GAIN)
    
    time.sleep(0.1)  # allow SDR to stabilize after initialization

    total_samples = int(SAMPLE_RATE * integration_time)
    collected = 0
    buffers = []

    while collected < total_samples:
        try:
            data = sdr.capture_data(CHUNK_SIZE)
            buffers.append(data)
            collected += len(data)
            time.sleep(SLEEP_TIME)
        except Exception as e:
            print("USB retry:", e)

    return np.concatenate(buffers)

input("next part")

# ---------- USER INPUT ----------
input_power_dbm = -40    # generator output in dBm
R = 50                   # system impedance in ohms
integration_time = 1     # seconds
# --------------------------------

# Convert generator power to true Vrms
P_watts = 10**((input_power_dbm - 30)/10)
V_rms_true = np.sqrt(P_watts * R)

print("Capturing calibration tone...")
data = safe_capture(integration_time)

I = np.real(data)
Q = np.imag(data)

# Measured digital RMS
V_meas = np.sqrt(np.mean(I**2 + Q**2))

# Calibration factor
K = V_rms_true / V_meas

print("True Vrms:", V_rms_true)
print("Measured digital RMS:", V_meas)
print("Voltage scale factor (V per digital unit):", K)

input("next part")

c = 3e8         # speed of light
L = 5.0         # length of test cable in meters

def measure_phase():
    """
    Returns the average phase of a short SDR capture.
    """
    data = safe_capture(0.5)   # short integration to reduce risk
    phasor = np.mean(data)
    return np.angle(phasor)

print("Measure SHORT cable...")
phi_short = measure_phase()

print("Measure LONG cable...")
phi_long = measure_phase()

# Phase difference (unwrapped)
delta_phi = np.angle(np.exp(1j*(phi_long - phi_short)))

# Calculate velocity in cable
v = (2 * np.pi * CENTER_FREQ * L) / delta_phi
VF = v / c

print("Phase difference (rad):", delta_phi)
print("Cable velocity (m/s):", v)
print("Velocity factor:", VF)

input("next part")

L = 5.0   # length of cable in meters

def measure_power_db():
    data = safe_capture(0.5)       # short capture
    power_lin = np.mean(np.abs(data)**2)
    return 10 * np.log10(power_lin)

print("Measure SHORT cable power...")
P_short = measure_power_db()

print("Measure LONG cable power...")
P_long = measure_power_db()

delta_P = P_short - P_long
alpha_db_per_m = delta_P / L

print("Short cable power (dB):", P_short)
print("Long cable power (dB):", P_long)
print("Total loss (dB):", delta_P)
print("Loss per meter (dB/m):", alpha_db_per_m)
