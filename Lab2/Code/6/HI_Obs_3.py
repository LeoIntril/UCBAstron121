import numpy as np
import ugradio
import os
import matplotlib.pyplot as plt

############################
# CONFIGURATION
############################

HI_FREQ = 1420.405751e6  # Hydrogen rest frequency [Hz]
c = 299792458            # Speed of light [m/s]

sample_rate = 1e6
nsamples = 2**14
nblocks = 8
gain = 40
freq_offset = 100e3      # LO offset for frequency switching

T_hot = 300  # Hot load temperature [K]
T_cold = 10  # Cold sky temperature [K]

outdir = "data"
os.makedirs(outdir, exist_ok=True)

############################
# HELPER FUNCTION
############################

def capture_one_shot(label, lo_freq):
    """
    Capture a single spectrum using a fresh SDR instance.
    Returns the raw data array.
    """
    input(f"Aim horn at {label} and press Enter...")

    # Initialize SDR
    sdr = ugradio.sdr.RtlSdr()
    sdr.sample_rate = sample_rate
    sdr.gain = gain
    sdr.direct = False
    sdr.center_freq = lo_freq

    # Capture data
    data = ugradio.sdr.capture_data(sdr, nsamples=nsamples, nblocks=nblocks)
    sdr.close()

    print(f"{label} capture complete.")
    return data

def compute_power(data):
    """Compute power spectrum from raw data"""
    fft = np.fft.fftshift(np.fft.fft(data[:nsamples]))
    power = np.abs(fft)**2
    return power

############################
# MAIN SCRIPT
############################

# 1️⃣ Cold sky
data_cold = capture_one_shot("COLD SKY", HI_FREQ)
spec_cold = compute_power(data_cold)

# 2️⃣ Hot load
data_hot = capture_one_shot("HOT LOAD", HI_FREQ)
spec_hot = compute_power(data_hot)

# 3️⃣ Y-factor and system temperature
P_cold = np.mean(spec_cold)
P_hot = np.mean(spec_hot)
Y = P_hot / P_cold
T_sys = (T_hot - Y*T_cold) / (Y - 1)
print(f"Y-factor: {Y:.4f}, Estimated system temperature: {T_sys:.2f} K")

# 4️⃣ Hydrogen observation
data_upper = capture_one_shot("TARGET HI UPPER LO", HI_FREQ - freq_offset)
spec_upper = compute_power(data_upper)

data_lower = capture_one_shot("TARGET HI LOWER LO", HI_FREQ + freq_offset)
spec_lower = compute_power(data_lower)

diff_spec = spec_upper - spec_lower
T_ant = T_sys * (diff_spec / P_cold)

# 5️⃣ Frequency and velocity axes
freqs = np.fft.fftshift(np.fft.fftfreq(nsamples, d=1/sample_rate))
rf_freqs = freqs + (HI_FREQ - freq_offset)  # reference to upper LO
velocity = c * (HI_FREQ - rf_freqs) / HI_FREQ / 1000  # km/s

# 6️⃣ Metadata
unix_time = ugradio.timing.unix_time()
lat = ugradio.nch.lat
lon = ugradio.nch.lon

# 7️⃣ Save all spectra and calibration
np.savez(f"{outdir}/hi_21cm_final.npz",
         freq_Hz=rf_freqs,
         velocity_kms=velocity,
         temperature_K=T_ant,
         s_cold=spec_cold,
         s_hot=spec_hot,
         spec_upper=spec_upper,
         spec_lower=spec_lower,
         diff_spec=diff_spec,
         T_sys_K=T_sys,
         Y_factor=Y,
         HI_rest_freq_Hz=HI_FREQ,
         lo_upper_Hz=HI_FREQ - freq_offset,
         lo_lower_Hz=HI_FREQ + freq_offset,
         sample_rate_Hz=sample_rate,
         gain_dB=gain,
         nsamples=nsamples,
         nblocks=nblocks,
         unix_time=unix_time,
         latitude_deg=lat,
         longitude_deg=lon)

print("Observation complete. Saved to hi_21cm_final.npz")

# 8️⃣ Optional: plot final calibrated spectrum
plt.figure()
plt.plot(velocity, T_ant)
plt.xlabel("Velocity (km/s)")
plt.ylabel("Antenna Temperature (K)")
plt.title("21-cm HI Spectrum")
plt.grid(True)
plt.show()
