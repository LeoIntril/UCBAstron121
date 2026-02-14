import numpy as np
import ugradio
from rtlsdr import RtlSdr
import os

############################
# CONFIGURATION
############################

HI_FREQ = 1420.405751e6
sample_rate = 1.0e6
nsamples = 2**14
nblocks = 8
gain = 40
freq_offset = 100e3

n_cal_integrations = 50
n_obs_integrations = 300

T_hot = 300
T_cold = 10

outdir = "data"
os.makedirs(outdir, exist_ok=True)

############################
# SETUP SDR
############################

sdr = RtlSdr()
sdr.sample_rate = sample_rate
sdr.gain = gain

############################
# INTEGRATION FUNCTION
############################

def integrate_spectrum(lo_freq, n_integrations):

    sdr.center_freq = lo_freq
    avg_spec = np.zeros(nsamples)

    for i in range(n_integrations):

        data = ugradio.sdr.capture_data(
            sdr,
            nsamples=nsamples,
            nblocks=nblocks
        )

        fft = np.fft.fftshift(np.fft.fft(data, axis=-1), axes=-1)
        power = np.abs(fft)**2
        spec = np.mean(power, axis=0)

        avg_spec += spec

    return avg_spec / n_integrations

############################
# ===== CALIBRATION =====
############################

print("\n=== INTENSITY CALIBRATION ===")

input("Aim horn at COLD SKY and press Enter...")
s_cold = integrate_spectrum(HI_FREQ, n_cal_integrations)

input("Place HOT blackbody in front of horn and press Enter...")
s_hot = integrate_spectrum(HI_FREQ, n_cal_integrations)

P_cold = np.mean(s_cold)
P_hot = np.mean(s_hot)

Y = P_hot / P_cold
T_sys = (T_hot - Y*T_cold) / (Y - 1)

print(f"Y-factor: {Y:.3f}")
print(f"System temperature: {T_sys:.2f} K")

############################
# ===== HYDROGEN OBS =====
############################

input("Aim horn at target and press Enter...")

lo_upper = HI_FREQ - freq_offset
lo_lower = HI_FREQ + freq_offset

spec_upper = integrate_spectrum(lo_upper, n_obs_integrations)
spec_lower = integrate_spectrum(lo_lower, n_obs_integrations)

diff_spec = spec_upper - spec_lower

T_ant = T_sys * (diff_spec / P_cold)

############################
# FREQUENCY AXIS
############################

freqs = np.fft.fftshift(
    np.fft.fftfreq(nsamples, d=1/sample_rate)
)
rf_freqs = freqs + lo_upper

############################
# ===== METADATA COLLECTION =====
############################

# Time
unix_time = ugradio.timing.unix_time()

# Observatory location
lat, lon = ugradio.nch.lat, ugradio.nch.lon

############################
# SAVE EVERYTHING
############################

np.savez(
    f"{outdir}/hi_calibrated_metadata.npz",

    # Spectral Data
    freq_Hz=rf_freqs,
    temperature_K=T_ant,
    diff_power=diff_spec,
    spec_upper=spec_upper,
    spec_lower=spec_lower,
    s_cold=s_cold,
    s_hot=s_hot,

    # Calibration
    T_sys_K=T_sys,
    Y_factor=Y,
    T_hot_K=T_hot,
    T_cold_K=T_cold,

    # Instrument Settings
    HI_rest_freq_Hz=HI_FREQ,
    lo_upper_Hz=lo_upper,
    lo_lower_Hz=lo_lower,
    sample_rate_Hz=sample_rate,
    gain_dB=gain,
    nsamples=nsamples,
    nblocks=nblocks,
    n_cal_integrations=n_cal_integrations,
    n_obs_integrations=n_obs_integrations,

    # Observatory Metadata
    unix_time=unix_time,
    latitude_deg=lat,
    longitude_deg=lon
)

print("\nObservation complete.")
print("Saved to data/hi_calibrated_metadata.npz")

sdr.close()
