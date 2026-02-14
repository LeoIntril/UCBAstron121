import numpy as np
import ugradio
from rtlsdr import RtlSdr
import os

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

outdir = "data"
os.makedirs(outdir, exist_ok=True)

############################
# SETUP SDR
############################

sdr = RtlSdr()
sdr.sample_rate = sample_rate
sdr.gain = gain
sdr.set_direct_sampling(0)

print("SDR configured")

############################
# INTEGRATION FUNCTION (uses ugradio properly)
############################

def integrate_spectrum(lo_freq, n_integrations):

    sdr.center_freq = lo_freq
    avg_spec = np.zeros(nsamples)

    for i in range(n_integrations):

        freqs, spec = ugradio.sdr.get_spectrum(
            sdr,
            nsamples=nsamples,
            nblocks=nblocks
        )

        avg_spec += spec

        if (i+1) % 20 == 0:
            print(f"  {i+1}/{n_integrations} integrations")

    return freqs, avg_spec / n_integrations

############################
# ===== INTENSITY CALIBRATION =====
############################

print("\n=== INTENSITY CALIBRATION ===")

input("Aim horn at COLD SKY, then press Enter...")
freqs, s_cold = integrate_spectrum(HI_FREQ, n_cal_integrations)

input("Place HOT absorber in front of horn, then press Enter...")
_, s_hot = integrate_spectrum(HI_FREQ, n_cal_integrations)

P_cold = np.mean(s_cold)
P_hot = np.mean(s_hot)

Y = P_hot / P_cold
T_sys = (T_hot - Y*T_cold) / (Y - 1)

print(f"\nY-factor: {Y:.4f}")
print(f"System Temperature: {T_sys:.2f} K")

############################
# ===== HYDROGEN OBSERVATION =====
############################

input("\nAim horn at Galactic plane target, then press Enter...")

lo_upper = HI_FREQ - freq_offset
lo_lower = HI_FREQ + freq_offset

print("\nObserving upper LO...")
freqs, spec_upper = integrate_spectrum(lo_upper, n_obs_integrations)

print("\nObserving lower LO...")
_, spec_lower = integrate_spectrum(lo_lower, n_obs_integrations)

############################
# FREQUENCY SWITCHING
############################

diff_spec = spec_upper - spec_lower

############################
# BANDPASS FLATTENING
############################

# Remove residual slope using low-order polynomial
x = np.arange(len(diff_spec))
poly = np.polyfit(x, diff_spec, 3)
baseline = np.polyval(poly, x)

flattened = diff_spec - baseline

############################
# TEMPERATURE CALIBRATION
############################

T_ant = T_sys * (flattened / P_cold)

############################
# RF FREQUENCY AXIS
############################

rf_freqs = freqs + lo_upper

############################
# DOPPLER VELOCITY AXIS
############################

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
    f"{outdir}/hi_grad_level.npz",

    # Spectral Data
    freq_Hz=rf_freqs,
    velocity_kms=velocity,
    temperature_K=T_ant,

    # Raw Components
    spec_upper=spec_upper,
    spec_lower=spec_lower,
    diff_power=diff_spec,
    flattened_power=flattened,
    s_cold=s_cold,
    s_hot=s_hot,

    # Calibration
    T_sys_K=T_sys,
    Y_factor=Y,
    T_hot_K=T_hot,
    T_cold_K=T_cold,

    # Instrument
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
print("Saved to data/hi_grad_level.npz")

sdr.close()
