import numpy as np
import ugradio
import os
import matplotlib.pyplot as plt

############################
# CONFIGURATION
############################

HI_FREQ = 1420.405751e6  # Hz
c = 299792458            # m/s

sample_rate = 1e6
nsamples = 2**14
nblocks = 8
gain = 40
freq_offset = 100e3

T_hot = 300  # K
T_cold = 10  # K

n_cal_integrations = 50
n_obs_integrations = 100
live_update_interval = 5

outdir = "data"
os.makedirs(outdir, exist_ok=True)

############################
# HELPER FUNCTION
############################

def capture_spectrum(label, lo_freq, n_integrations):
    """
    Capture multiple blocks using a fresh SDR for each integration.
    Returns the averaged power spectrum.
    """
    avg_spec = np.zeros(nsamples)
    freqs = np.fft.fftshift(np.fft.fftfreq(nsamples, d=1/sample_rate))
    rf_freqs = freqs + lo_freq

    # Setup live plot
    plt.ion()
    fig, ax = plt.subplots()
    line, = ax.plot(rf_freqs/1e6, avg_spec)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power")
    ax.set_title(label)

    for i in range(n_integrations):
        # Initialize SDR fresh to avoid freezing
        sdr = ugradio.sdr.RtlSdr()
        sdr.sample_rate = sample_rate
        sdr.gain = gain
        sdr.direct = False
        sdr.center_freq = lo_freq

        data = ugradio.sdr.capture_data(sdr, nsamples=nsamples, nblocks=nblocks)
        sdr.close()

        fft = np.fft.fftshift(np.fft.fft(data[:nsamples]))
        power = np.abs(fft)**2
        avg_spec += power

        if (i+1) % live_update_interval == 0:
            line.set_ydata(avg_spec / (i+1))
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)
            print(f"{label}: {i+1}/{n_integrations} integrations")

    plt.ioff()
    plt.show()
    return avg_spec / n_integrations

############################
# MAIN SCRIPT
############################

# 1️⃣ Cold Sky
input("Aim horn at COLD SKY and press Enter...")
s_cold = capture_spectrum("Cold Sky", HI_FREQ, n_cal_integrations)

# 2️⃣ Hot Load
input("Place HOT load in front of horn and press Enter...")
s_hot = capture_spectrum("Hot Load", HI_FREQ, n_cal_integrations)

# 3️⃣ Compute Y-factor and T_sys
P_cold = np.mean(s_cold)
P_hot = np.mean(s_hot)
Y = P_hot / P_cold
T_sys = (T_hot - Y*T_cold) / (Y - 1)
print(f"Y-factor: {Y:.4f}, System temperature: {T_sys:.2f} K")

# 4️⃣ Hydrogen Observation
input("Aim horn at target and press Enter...")
lo_upper = HI_FREQ - freq_offset
lo_lower = HI_FREQ + freq_offset

spec_upper = capture_spectrum("HI Upper LO", lo_upper, n_obs_integrations)
spec_lower = capture_spectrum("HI Lower LO", lo_lower, n_obs_integrations)

diff_spec = spec_upper - spec_lower
T_ant = T_sys * (diff_spec / P_cold)

# Frequency and velocity axes
freqs = np.fft.fftshift(np.fft.fftfreq(nsamples, d=1/sample_rate))
rf_freqs = freqs + lo_upper
velocity = c * (HI_FREQ - rf_freqs) / HI_FREQ / 1000  # km/s

# Metadata
unix_time = ugradio.timing.unix_time()
lat = ugradio.nch.lat
lon = ugradio.nch.lon

# Save results
np.savez(f"{outdir}/hi_21cm_final.npz",
         freq_Hz=rf_freqs,
         velocity_kms=velocity,
         temperature_K=T_ant,
         s_cold=s_cold,
         s_hot=s_hot,
         spec_upper=spec_upper,
         spec_lower=spec_lower,
         diff_spec=diff_spec,
         T_sys_K=T_sys,
         Y_factor=Y,
         HI_rest_freq_Hz=HI_FREQ,
         lo_upper_Hz=lo_upper,
         lo_lower_Hz=lo_lower,
         sample_rate_Hz=sample_rate,
         gain_dB=gain,
         nsamples=nsamples,
         nblocks=nblocks,
         n_cal_integrations=n_cal_integrations,
         n_obs_integrations=n_obs_integrations,
         unix_time=unix_time,
         latitude_deg=lat,
         longitude_deg=lon)

print("Observation complete. Saved to hi_21cm_final.npz")

# Optional: plot final calibrated spectrum
plt.figure()
plt.plot(velocity, T_ant)
plt.xlabel("Velocity (km/s)")
plt.ylabel("Antenna Temperature (K)")
plt.title("21-cm HI Spectrum")
plt.show()
