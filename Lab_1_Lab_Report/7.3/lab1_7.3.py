import numpy as np
import ugradio.sdr

# -------------------------
# SDR settings
# -------------------------
fs = 1_000_000      # Hz
N  = 16_384

sdr = ugradio.sdr.SDR(sample_rate=fs)

# Flush stale samples
_ = sdr.read_samples(4096)

# -------------------------
# Capture IF
# -------------------------
x = sdr.read_samples(N).astype(float)

np.savez(
    "IF_from_KeysightLO10MHz_RF9p8MHz.npz",
    data=x,
    sample_rate=fs,
    N=N
)

print("Saved IF from Keysight LO path:", x.shape)



import numpy as np
import ugradio.sdr

# -------------------------
# SDR capture settings
# -------------------------
fs = 1_000_000        # sampling rate [Hz]
N  = 16_384           # number of samples

sdr = ugradio.sdr.SDR(sample_rate=fs)

# Flush stale samples
_ = sdr.read_samples(4096)

# -------------------------
# Capture IF signal
# -------------------------
x = sdr.read_samples(N).astype(float)

# -------------------------
# Save to file
# -------------------------
np.savez(
    "IF_from_KeysightN9310A_10MHz_delayedLO_RF9p8MHz_CH2.npz",
    data=x,
    sample_rate=fs,
    N=N
)

print("Saved Channel 2 (Q) IF capture:", x.shape)


import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Load Channel 1 data
# -------------------------
d = np.load("IF_from_KeysightN9310A_10MHz_RF9p8MHz.npz")
x  = d["data"]
fs = float(d["sample_rate"])

# -------------------------
# Time axis
# -------------------------
t = np.arange(len(x)) / fs

# -------------------------
# FFT (no custom packages)
# -------------------------
X = np.fft.fft(x)
freqs = np.fft.fftfreq(len(x), d=1/fs)
P = np.abs(X)**2

# Shift for centered spectrum
freqs = np.fft.fftshift(freqs)
P = np.fft.fftshift(P)

# -------------------------
# Measured IF frequency
# -------------------------
f_measured = abs(freqs[np.argmax(P)])

print("Sampling rate:", fs, "Hz")
print("Measured IF frequency:", f_measured, "Hz")

# -------------------------
# Plots
# -------------------------
plt.figure(figsize=(10,4))

# Time-domain (zoomed)
plt.subplot(1,2,1)
plt.plot(t[:300], x[:300])
plt.xlabel("Time (s)")
plt.ylabel("Voltage (arb)")
plt.title("Channel 1 IF (time domain)")

# Frequency-domain
plt.subplot(1,2,2)
plt.plot(freqs, P)
plt.xlim(-500_000, 500_000)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power |FFT|² (arb)")
plt.title("Channel 1 IF Spectrum")

plt.tight_layout()
plt.show()
