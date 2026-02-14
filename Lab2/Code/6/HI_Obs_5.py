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
nblocks_cal = 8          # Blocks for calibration (short)
nblocks_obs = 128        # Longer integration for HI line detection
gain = 40
sample_rate = 1e6
freq_offset = 100e3      # LO offset

T_hot = 300  # Hot load temperature [K]
T_cold = 10  # Cold sky temperature [K]

live_update_interval = 4  # update plot every N blocks

outdir = "data"
os.makedirs(outdir, exist_ok=True)

############################
# HELPER FUNCTION
############################

def capture(label, center_freq, nblocks, live=False):
    """Capture voltages and power spectrum with ugradio SDR."""
    input(f"Aim horn at {label} and press Enter...")

    sdr = ugradio.sdr.SDR(direct=False)
    sdr.sample_rate = sample_rate
    sdr.gain = gain
    sdr.center_freq = center_freq

    # Initialize accumulation
    avg_power = np.zeros(nsamples * nblocks, dtype=np.float64)

    # Live plot setup
    if live:
        plt.ion()
        fig, ax = plt.subplots()
        freqs = np.fft.fftshift(np.fft.fftfreq(nsamples * nblocks, d=1/sample_rate))
        rf_freqs = freqs + center_freq
        line, = ax.plot(rf_freqs, avg_power)
        ax.set_xlabel("Frequency [Hz]")
        ax.set_ylabel("Power")
        ax.set_title(f"Live: {label}")

    # Capture each block individually
    for b in range(nblocks):
        result = ugradio.sdr.capture_data(sdr, nsamples=nsamples, nblocks=1)

        # Extract voltages
        if isinstance(result, dict):
            if 'data' in result:
                voltages = np.array(result['data'], dtype=np.complex64).flatten()
            elif 'samples' in result:
                voltages = np.array(result['samples'], dtype=np.complex64).flatten()
            else:
                first_array = next(v for v in result.values() if isinstance(v, (list, np.ndarray)))
                voltages = np.array(first_array, dtype=np.complex64).flatten()
        else:
            voltages = np.array(result, dtype=np.complex64).flatten()

        # FFT and power
        fft = np.fft.fftshift(np.fft.fft(voltages))
        power = np.abs(fft)**2
        avg_power[b*nsamples:(b+1)*nsamples] = power

        # Update live plot
        if live and ((b+1) % live_update_interval == 0 or b == nblocks-1):
            line.set_ydata(avg_power)
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)
            print(f"{label}: Completed {b+1}/{nblocks} blocks")

    if live:
        plt.ioff()
        plt.show()

    print(f"{label} capture complete. Data length: {len(avg_power)}")
    sdr.close()
    return avg_power

############################
# 1️⃣ Cold Sky Calibration
############################
spec_cold = capture("COLD SKY", HI_FREQ, nblocks=nblocks_cal)

############################
# 2️⃣ Hot Load Calibration
############################
spec_hot = capture("HOT LOAD", HI_FREQ, nblocks=nblocks_cal)

# Compute Y-factor and system temperature
P_cold = np.mean(spec_cold)
P_hot = np.mean(spec_hot)
Y = P_hot / P_cold
T_sys = (T_hot - Y*T_cold) / (Y - 1)
print(f"Y-factor: {Y:.4f}, Estimated system temperature: {T_sys:.2f} K")

############################
# 3️⃣ Hydrogen Observation with Live Plot
############################
spec_upper = capture("HI TARGET UPPER LO", HI_FREQ - freq_offset, nblocks=nblocks_obs, live=True)
spec_lower = capture("HI TARGET LOWER LO", HI_FREQ + freq_offset, nblocks=nblocks_obs, live=True)

diff_spec = spec_upper - spec_lower
T_ant = T_sys * (diff_spec / P_cold)

############################
# 4️⃣ Frequency and velocity axes
############################
len_fft = len(T_ant)
freqs = np.fft.fftshift(np.fft.fftfreq(len_fft, d=1/sample_rate))
rf_freqs = freqs + (HI_FREQ - freq_offset)
velocity = c * (HI_FREQ - rf_freqs) / HI_FREQ / 1000  # km/s

############################
# 5️⃣ Metadata
############################
unix_time = ugradio.timing.unix_time()
lat = ugradio.nch.lat
lon = ugradio.nch.lon

############################
# 6️⃣ Save results
############################
np.savez(os.path.join(outdir, "hi_21cm_final_sdr_live.npz"),
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
         nblocks_cal=nblocks_cal,
         nblocks_obs=nblocks_obs,
         unix_time=unix_time,
         latitude_deg=lat,
         longitude_deg=lon
         )

print("Observation complete. Saved to hi_21cm_final_sdr_live.npz")

############################
# 7️⃣ Plot final spectrum
plt.figure()
plt.plot(velocity, T_ant)
plt.xlabel("Velocity (km/s)")
plt.ylabel("Antenna Temperature (K)")
plt.title("21-cm HI Spectrum")
plt.grid(True)
plt.show()
