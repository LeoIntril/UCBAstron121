import numpy as np
import ugradio
import os
import matplotlib.pyplot as plt

############################
# CONFIGURATION
############################

HI_FREQ = 1420.405751e6      # Hydrogen rest frequency (Hz)
c = 299792458                # Speed of light (m/s)

sample_rate = 1.0e6
nsamples = 2**14
nblocks = 8
gain = 40
freq_offset = 100e3

n_cal_integrations = 50
n_obs_integrations = 300

T_hot = 300      # K
T_cold = 10      # K

live_update_interval = 5  # Update plot every 5 integrations

outdir = "data"
os.makedirs(outdir, exist_ok=True)

############################
# SETUP SDR
############################

sdr = ugradio.sdr.RtlSdr()
sdr.sample_rate = sample_rate
sdr.gain = gain
sdr.direct = False

print("SDR configured:")
print(f"  Sample rate: {sample_rate/1e6:.2f} MS/s")
print(f"  FFT size: {nsamples}")
print(f"  Gain: {gain} dB")
print("  Direct sampling (direct = False) confirmed")

############################
# HELPER FUNCTION TO CONVERT CAPTURE TO NUMPY
############################

def capture_to_array(sdr, nsamples, nblocks):
    """
    Convert the rtlsdraio object returned by capture_data() into a NumPy array.
    """
    raw = ugradio.sdr.capture_data(sdr, nsamples=nsamples, nblocks=nblocks)
    # latest ugradio object can be iterated block by block
    data = np.array([block for block in raw])
    return data

############################
# INTEGRATION FUNCTION WITH LIVE PLOT
############################

def integrate_spectrum_live(lo_freq, n_integrations, label="Spectrum"):
    sdr.center_freq = lo_freq
    avg_spec = np.zeros(nsamples)

    plt.ion()
    fig, ax = plt.subplots()
    freqs = np.fft.fftshift(np.fft.fftfreq(nsamples, d=1/sample_rate))
    rf_freqs = freqs + lo_freq
    line, = ax.plot(rf_freqs, avg_spec)
    ax.set_xlabel("Frequency [Hz]")
    ax.set_ylabel("Power")
    ax.set_title(label)

    for i in range(n_integrations):
        data = capture_to_array(sdr, nsamples, nblocks)
        # FFT along last axis
        fft = np.fft.fftshift(np.fft.fft(data, axis=-1), axes=-1)
        power = np.abs(fft)**2
        spec = np.mean(power, axis=0)
        avg_spec += spec

        if (i+1) % live_update_interval == 0:
            line.set_ydata(avg_spec / (i+1))
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)
            print(f"{label}: Completed {i+1}/{n_integrations} integrations")

    plt.ioff()
    plt.show()
    return avg_spec / n_integrations

############################
# CALIBRATION
############################

print("\n=== INTENSITY CALIBRATION ===")
input("Aim horn at COLD SKY, then press Enter...")
s_cold = integrate_spectrum_live(HI_FREQ, n_cal_integrations, label="Cold Sky")

input("Place HOT blackbody in front of horn, then press Enter...")
s_hot = integrate_spectrum_live(HI_FREQ, n_cal_integrations, label="Hot Load")

P_cold = np.mean(s_cold)
P_hot = np.mean(s_hot)
Y = P_hot / P_cold
T_sys = (T_hot - Y*T_cold) / (Y - 1)

print(f"\nY-factor: {Y:.4f}")
print(f"Estimated system temperature: {T_sys:.2f} K")

############################
# HYDROGEN OBSERVATION
############################

input("\nAim horn at target (galactic plane), then press Enter...")
lo_upper = HI_FREQ - freq_offset
lo_lower = HI_FREQ + freq_offset

print("\nObserving upper LO...")
spec_upper = integrate_spectrum_live(lo_upper, n_obs_integrations, label="HI Upper LO")

print("\nObserving lower LO...")
spec_lower = integrate_spectrum_live(lo_lower, n_obs_integrations, label="HI Lower LO")

diff_spec = spec_upper - spec_lower
T_ant = T_sys * (diff_spec / P_cold)

############################
# FREQUENCY AND VELOCITY AXIS
############################

freqs = np.fft.fftshift(np.fft.fftfreq(nsamples, d=1/sample_rate))
rf_freqs = freqs + lo_upper
velocity = c * (HI_FREQ - rf_freqs) / HI_FREQ / 1000  # km/s

############################
# METADATA
############################

unix_time = ugradio.timing.unix_time()
lat = ugradio.nch.lat
lon = ugradio.nch.lon

############################
# SAVE RESULTS
############################

np.savez(
    f"{outdir}/hi_ugradio_live.npz",
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
    T_hot_K=T_hot,
    T_cold_K=T_cold,
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
    longitude_deg=lon
)

print("\nObservation complete.")
print("Saved to data/hi_ugradio_live.npz")
