import numpy as np
import ugradio

# ---------- USER SETTINGS ----------
center_freq = 1420e6
sample_rate = 2.4e6
gain = 20              # SDR gain setting
integration_time = 2   # seconds
input_power_dbm = -40  # Generator output power
R = 50                 # system impedance (ohms)
# -----------------------------------

# Convert dBm to true Vrms
P_watts = 10**((input_power_dbm - 30)/10)
V_rms_true = np.sqrt(P_watts * R)

# Setup SDR
sdr = ugradio.sdr.SDR(freq=center_freq,
                      samp_rate=sample_rate,
                      gain=gain)

print("Collecting data...")
data = sdr.capture_data(int(sample_rate * integration_time))

I = np.real(data)
Q = np.imag(data)

# Measured digital RMS
V_meas = np.sqrt(np.mean(I**2 + Q**2))

# Calibration factor
K = V_rms_true / V_meas

print("True Vrms:", V_rms_true)
print("Measured digital RMS:", V_meas)
print("Voltage scale factor (Volts per digital unit):", K)
