import numpy as np
import ugradio
from rtlsdr import RtlSdr

############################
# CONFIGURATION PARAMETERS
############################

HI_FREQ = 1420.405751e6   # HI rest frequency (Hz)

sample_rate = 1.2e6      # Hz (safe for USB + RPi)
gain = 40                # dB (adjust for your system)
nblocks = 16
nsamples = 2**18         # large for noise reduction

# Frequency switching offset (Hz)
# Keeps line well inside analog passband
freq_offset = 250e3      # 100 kHz

############################
# SET UP SDR
############################

sdr = RtlSdr()

sdr.sample_rate = sample_rate
sdr.gain = gain
sdr.center_freq = HI_FREQ  # temporary placeholder

print("SDR configured:")
print(f"Sample rate: {sample_rate/1e6:.2f} MS/s")
print(f"Gain: {gain} dB")

############################
# HELPER FUNCTION
############################

def capture_spectrum(
    sdr,
    lo_freq,
    label="",
    ntrials=4,
    outdir="data"
):
    """
    Capture multiple spectra at a given LO frequency.
    
    Saves one NPZ per trial containing:
      - raw voltage data
      - power spectrum
      - LO frequency
      - metadata
    
    Returns:
      spectra: (ntrials, nsamples) array
      avg_spec: averaged spectrum
    """

    sdr.center_freq = lo_freq
    print(f"\nTuned LO to {lo_freq/1e6:.6f} MHz ({label})")

    spectra = []

    for trial in range(ntrials):
        print(f"  Trial {trial+1}/{ntrials}")

        data = ugradio.sdr.capture_data(
            sdr,
            nsamples=nsamples,
            nblocks=nblocks
        )

        # FFT and power spectrum
        fft = np.fft.fftshift(np.fft.fft(data, axis=-1), axes=-1)
        power = np.abs(fft)**2

        # Average over blocks
        spec = np.mean(power, axis=0)
        spectra.append(spec)

        # Save trial
        fname = (
            f"{outdir}/"
            f"{label}_"
            f"{int(lo_freq/1e3)}kHz_"
            f"trial{trial:02d}.npz"
        )

        np.savez(
            fname,
            raw_data=data,
            spectrum=spec,
            lo_freq=lo_freq,
            sample_rate=sample_rate,
            nsamples=nsamples,
            nblocks=nblocks,
            label=label,
            trial=trial
        )

        print(f"    Saved {fname}")

    spectra = np.array(spectra)
    avg_spec = np.mean(spectra, axis=0)

    return spectra, avg_spec


############################
# IN-BAND FREQUENCY SWITCHING
############################

# HI line in upper half of spectrum
lo_upper = HI_FREQ - freq_offset

# HI line in lower half of spectrum
lo_lower = HI_FREQ + freq_offset

ntrials = 6

# HI in upper half
lo_upper = HI_FREQ - freq_offset
specs_upper, avg_upper = capture_spectrum(
    sdr,
    lo_upper,
    label="hi_upper",
    ntrials=ntrials
)

# HI in lower half
lo_lower = HI_FREQ + freq_offset
specs_lower, avg_lower = capture_spectrum(
    sdr,
    lo_lower,
    label="hi_lower",
    ntrials=ntrials
)


############################
# FREQUENCY AXIS
############################

freqs = np.fft.fftshift(
    np.fft.fftfreq(nsamples, d=1/sample_rate)
)

############################
# INTENSITY CALIBRATION
############################

print("Calibration: point horn at cold sky")
input("Point horn at cold sky and press Enter...")
cold_specs, s_cold = capture_spectrum(
    sdr,
    lo_upper,
    label="cold_sky",
    ntrials=3
)

print("Calibration: point horn at people / warm load")
input("Point horn at people / warm load and press Enter...")
cal_specs, s_cal = capture_spectrum(
    sdr,
    lo_upper,
    label="warm_load",
    ntrials=3
)

print("Observation complete. Data saved to hi_observation.npz")

############################
# CLEAN UP
############################

sdr.close()
