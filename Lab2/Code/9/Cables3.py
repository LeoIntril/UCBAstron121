import numpy as np
import ugradio
import time

# ---------- SDR SETTINGS ----------
CENTER_FREQ = 1420e6
SAMPLE_RATE = 250_000  # NESDR SMArt valid
GAIN = 20
CHUNK_SIZE = 4096       # smaller = more stable
SLEEP_TIME = 0.005      # let USB buffer drain
BLOCK_TIME = 0.1        # seconds per short block
# -----------------------------------

def block_capture(total_integration_time):
    """
    Capture samples in small blocks to avoid Pi USB hang.
    Returns concatenated IQ array.
    """
    sdr = ugradio.sdr.SDR(center_freq=CENTER_FREQ,
                          sample_rate=SAMPLE_RATE,
                          gain=GAIN)
    time.sleep(0.1)  # allow SDR to stabilize

    num_blocks = int(np.ceil(total_integration_time / BLOCK_TIME))
    IQ_blocks = []

    for b in range(num_blocks):
        block_samples = int(SAMPLE_RATE * BLOCK_TIME)
        collected = 0
        data_list = []

        while collected < block_samples:
            try:
                d = sdr.capture_data(CHUNK_SIZE)
                data_list.append(d)
                collected += len(d)
                time.sleep(SLEEP_TIME)
            except Exception as e:
                print("USB retry:", e)

        block_data = np.concatenate(data_list)
        IQ_blocks.append(block_data)

    return np.concatenate(IQ_blocks)

input("next part")

# User input
input_power_dbm = -40  # generator output
R = 50                  # ohms
integration_time = 1    # total seconds

# Convert dBm → Vrms
P_watts = 10**((input_power_dbm - 30)/10)
V_rms_true = np.sqrt(P_watts * R)

print("Capturing calibration tone...")
data = block_capture(integration_time)

I = np.real(data)
Q = np.imag(data)

V_meas = np.sqrt(np.mean(I**2 + Q**2))
K = V_rms_true / V_meas

print("True Vrms:", V_rms_true)
print("Measured digital RMS:", V_meas)
print("Calibration factor (V per digital unit):", K)

input("next part")

c = 3e8         # speed of light
L = 5.0         # cable length in meters

def measure_phase():
    data = block_capture(0.5)  # short integration for safety
    phasor = np.mean(data)
    return np.angle(phasor)

print("Measure SHORT cable...")
phi_short = measure_phase()

print("Measure LONG cable...")
phi_long = measure_phase()

delta_phi = np.angle(np.exp(1j*(phi_long - phi_short)))

v = (2 * np.pi * CENTER_FREQ * L) / delta_phi
VF = v / c

print("Phase difference (rad):", delta_phi)
print("Cable velocity (m/s):", v)
print("Velocity factor:", VF)

input("next part")

L = 5.0  # cable length in meters

def measure_power_db():
    data = block_capture(0.5)  # short integration
    power_lin = np.mean(np.abs(data)**2)
    return 10 * np.log10(power_lin)

print("Measure SHORT cable power...")
P_short = measure_power_db()

input("long cable")
print("Measure LONG cable power...")
P_long = measure_power_db()

delta_P = P_short - P_long
alpha_db_per_m = delta_P / L

print("Short power (dB):", P_short)
print("Long power (dB):", P_long)
print("Total loss (dB):", delta_P)
print("Loss per meter (dB/m):", alpha_db_per_m)
