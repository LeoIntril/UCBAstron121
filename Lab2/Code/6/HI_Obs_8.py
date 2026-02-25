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

outdir = "data"
os.makedirs(outdir, exist_ok=True)

############################
# HELPER FUNCTION
############################

def capture(label, center_freq, nblocks=1, return_blocks=False):
    """
    Capture voltages and power spectrum with ugradio SDR.
    
    If return_blocks=True, returns an array of shape (nblocks, nsamples)
    without averaging, suitable for frequency switching.
    """
    input(f"Aim horn at {label} and press Enter...")

    sdr = ugradio.sdr.SDR(direct=False)
    sdr.sample_rate = sample_rate
    sdr.gain = gain
    sdr.center_freq = center_freq

    result = ugradio.sdr.capture_data(sdr, nsamples=nsamples, nblocks=nblocks)

    # Extract voltages robustly (same as before)
    if isinstance(result, dict):
        if 'samples' in result:
            voltages = np.array(result['samples'])
        elif 'data' in result:
            raw = np.array(result['data'], dtype=np.float32).flatten()
            I = raw[0::2]
            Q = raw[1::2]
            voltages = I + 1j*Q
        else:
            first_array = next(v for v in result.values() if isinstance(v, (list, np.ndarray)))
            voltages = np.array(first_array, dtype=np.complex64)
    else:
        voltages = np.array(result, dtype=np.complex64)

    # Ensure shape
    if voltages.ndim == 1:
        voltages = voltages.reshape(nblocks, nsamples)
    elif voltages.shape != (nblocks, nsamples):
        voltages = voltages.flatten()
        voltages = voltages.reshape(nblocks, nsamples)

    # Compute averaged power only if requested
    if return_blocks:
        # Return raw blocks, each FFT can be computed later
        sdr.close()
        return voltages
    else:
        power_blocks = []
        for block in voltages:
            fft_block = np.fft.fft(block)
            power_blocks.append(np.abs(np.fft.fftshift(fft_block))**2)
        power = np.mean(power_blocks, axis=0)
        sdr.close()
        return voltages, power

############################
# 1 Cold Sky Calibration
############################
data_cold, spec_cold = capture("COLD SKY", HI_FREQ, nblocks=nblocks_cal)

############################
# 2 Hot Load Calibration
############################
data_hot, spec_hot = capture("HOT LOAD", HI_FREQ, nblocks=nblocks_cal)

# Compute Y-factor and system temperature
P_cold = np.mean(spec_cold)
P_hot = np.mean(spec_hot)
Y = P_hot / P_cold
T_sys = (T_hot - Y*T_cold) / (Y - 1)
print(f"Y-factor: {Y:.4f}, Estimated system temperature: {T_sys:.2f} K")

############################
# 3 Hydrogen Observation with Frequency Switching (Save raw + power)
############################

############################
# 3 Hydrogen Observation with Frequency Switching (Save raw + power)
############################

# --- Prepare lists to store block data ---
upper_blocks_volt = []   # raw complex voltages
upper_blocks_power = []  # power spectra
lower_blocks_volt = []
lower_blocks_power = []

sdr = ugradio.sdr.SDR(direct=False)
sdr.sample_rate = sample_rate
sdr.gain = gain

input("Aim horn at HI target and press Enter to begin frequency-switched observation...")

for i in range(nblocks_obs):
    # Alternate LO
    if i % 2 == 0:
        center_freq = HI_FREQ - freq_offset  # upper LO = on-line
        current_label = f"HI TARGET Upper LO, block {i+1}"
        is_upper = True
    else:
        center_freq = HI_FREQ + freq_offset  # lower LO = off-line
        current_label = f"HI TARGET Lower LO, block {i+1}"
        is_upper = False

    print(f"Capturing {current_label}")

    # --- Capture one raw block, preserving block-level voltages ---
    voltages_block = capture(current_label, center_freq, nblocks=1, return_blocks=True)

    # --- Compute FFT + power spectrum for this block ---
    fft_block = np.fft.fft(voltages_block[0])
    power_block = np.abs(np.fft.fftshift(fft_block))**2

    # --- Store raw voltages and power ---
    if is_upper:
        upper_blocks_volt.append(voltages_block[0])
        upper_blocks_power.append(power_block)
    else:
        lower_blocks_volt.append(voltages_block[0])
        lower_blocks_power.append(power_block)

sdr.close()

# --- Convert lists to arrays ---
upper_blocks_volt = np.array(upper_blocks_volt)
upper_blocks_power = np.array(upper_blocks_power)
lower_blocks_volt = np.array(lower_blocks_volt)
lower_blocks_power = np.array(lower_blocks_power)

# --- Average the power spectra for each LO (after block-level stacking) ---
spec_upper_avg = np.mean(upper_blocks_power, axis=0)
spec_lower_avg = np.mean(lower_blocks_power, axis=0)

# --- Build baseband frequency axis ---
len_fft = len(spec_upper_avg)
freqs = np.fft.fftshift(np.fft.fftfreq(len_fft, d=1/sample_rate))  # Hz

# --- Build RF axis reference for both LOs ---
rf_upper = freqs + (HI_FREQ - freq_offset)
rf_lower = freqs + (HI_FREQ + freq_offset)

# --- Interpolate off-line (lower LO) onto on-line (upper LO) RF grid ---
from scipy.interpolate import interp1d
interp_lower = interp1d(rf_lower, spec_lower_avg, bounds_error=False, fill_value=0)
spec_lower_on_upper = interp_lower(rf_upper)

# --- Frequency-switched stacked spectrum (on-line minus off-line) ---
diff_spec_stack = spec_upper_avg - spec_lower_on_upper

# --- Convert to antenna temperature ---
T_ant = T_sys * (diff_spec_stack / P_cold)

# --- Velocity axis (km/s) ---
velocity = c * (HI_FREQ - rf_upper) / HI_FREQ / 1000

# --- Velocity and Frequency axis ---
len_fft = len(T_ant)
freqs = np.fft.fftshift(np.fft.fftfreq(len_fft, d=1/sample_rate))
rf_freqs = freqs + (HI_FREQ - freq_offset)  # reference to upper LO
velocity = c * (HI_FREQ - rf_freqs) / HI_FREQ / 1000  # km/s

############################
# 4 Metadata
############################
unix_time = ugradio.timing.unix_time()
lat = ugradio.nch.lat
lon = ugradio.nch.lon

############################
# 5 Save results
############################
np.savez(os.path.join(outdir, "hi_21cm_final_sdr_longint_freqswitch.npz"),
         freq_Hz=rf_upper,
         velocity_kms=velocity,
         temperature_K=T_ant,
         data_cold=data_cold,
         s_cold=spec_cold,
         data_hot=data_hot,
         s_hot=spec_hot,
         upper_volt=upper_blocks_volt,
         upper_power=upper_blocks_power,
         lower_volt=lower_blocks_volt,
         lower_power=lower_blocks_power,
         spec_upper_avg=spec_upper_avg,
         spec_lower_avg=spec_lower_avg,
         diff_spec=diff_spec_stack,
         T_sys_K=T_sys,
         Y_factor=Y,
         HI_rest_freq_Hz=HI_FREQ,
         lo_upper_Hz=HI_FREQ - freq_offset,
         lo_lower_Hz=HI_FREQ + freq_offset,
         nsamples=nsamples,
         nblocks_cal=nblocks_cal,
         nblocks_obs=nblocks_obs,
         unix_time=ugradio.timing.unix_time(),
         latitude_deg=ugradio.nch.lat,
         longitude_deg=ugradio.nch.lon
         )

print("Frequency-switched observation complete and saved.")

############################
# 5 Plot frequency-switched spectrum
plt.figure(figsize=(10,5))
plt.plot(velocity, T_ant)
plt.xlabel("Velocity (km/s)")
plt.ylabel("Antenna Temperature (K)")
plt.title("21-cm HI Spectrum (Frequency-Switched)")
plt.grid(True)
plt.show()
