import numpy as np
import ugradio
import os
import matplotlib.pyplot as plt

############################
# CONFIGURATION
############################

HI_FREQ = 1420.405751e6  # Hydrogen rest frequency [Hz]
c = 299792458            # Speed of light [m/s]

nsamples = 2**14
nblocks = 8
gain = 40
sample_rate = 1e6
freq_offset = 100e3      # LO offset

T_hot = 300  # Hot load temperature [K]
T_cold = 10  # Cold sky temperature [K]

outdir = "data"
os.makedirs(outdir, exist_ok=True)

############################
# HELPER FUNCTIONS
############################

def capture(label, center_freq):
    input(f"Aim horn at {label} and press Enter...")

    # Initialize SDR
    sdr = ugradio.sdr.SDR(direct=False)
    sdr.sample_rate = sample_rate
    sdr.gain = gain
    sdr.center_freq = center_freq

    # Capture data (dict)
    result = ugradio.sdr.capture_data(sdr, nsamples=nsamples, nblocks=nblocks)

    # Extract voltage array
    voltages = np.array(result['data'], dtype=np.complex64).flatten()

    # FFT and power spectrum
    fft = np.fft.fftshift(np.fft.fft(voltages))
    power = np.abs(fft)**2

    print(f"{label} capture complete. Data length: {len(voltages)}")
    sdr.close()
    return voltages, power


############################
# 1️⃣ Cold Sky
data_cold, spec_cold = capture("COLD SKY", HI_FREQ)

############################
# 2️⃣ Hot Load
data_hot, spec_hot = capture("HOT LOAD", HI_FREQ)

############################
# 3️⃣ Compute Y-factor and system temperature
P_cold = np.mean(spec_cold)
P_hot = np.mean(spec_hot)
Y = P_hot / P_cold
T_sys = (T_hot - Y*T_cold) / (Y - 1)
print(f"Y-factor: {Y:.4f}, Estimated system temperature: {T_sys:.2f} K")

############################
# 4️⃣ Hydrogen Observation
data_upper, spec_upper = capture("HI TARGET UPPER LO", HI_FREQ - freq_offset)
data_lower, spec_lower = capture("HI TARGET LOWER LO", HI_FREQ + freq_offset)

diff_spec = spec_upper - spec_lower
T_ant = T_sys * (diff_spec / P_cold)

############################
# 5️⃣ Frequency and velocity axes
freqs = np.fft.fftshift(np.fft.fftfreq(nsamples, d=1/sample_rate))
rf_freqs = freqs + (HI_FREQ - freq_offset)  # reference to upper LO
velocity = c * (HI_FREQ - rf_freqs) / HI_FREQ / 1000  # km/s

############################
# 6️⃣ Metadata
unix_time = ugradio.timing.unix_time()
lat = ugradio.nch.lat
lon = ugradio.nch.lon

############################
# 7️⃣ Save results
np.savez(os.path.join(outdir, "hi_21cm_final_sdr.npz"),
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
         nsamples=nsamples,
         nblocks=nblocks,
         unix_time=unix_time,
         latitude_deg=lat,
         longitude_deg=lon
         )

print("Observation complete. Saved to hi_21cm_final_sdr.npz")

############################
# 8️⃣ Plot final spectrum
plt.figure()
plt.plot(velocity, T_ant)
plt.xlabel("Velocity (km/s)")
plt.ylabel("Antenna Temperature (K)")
plt.title("21-cm HI Spectrum")
plt.grid(True)
plt.show()
