import numpy as np
import ugradio
from rtlsdr import RtlSdr
import matplotlib.pyplot as plt
import time

############################
# CONFIGURATION
############################

HI_FREQ = 1420.405751e6
sample_rate = 1.0e6
nsamples = 2**14
nblocks = 8
gain = 40
freq_offset = 100e3

update_interval = 10   # updates before plot refresh
smooth_width = 7

############################
# SETUP SDR
############################

sdr = RtlSdr()
sdr.sample_rate = sample_rate
sdr.gain = gain

############################
# FREQUENCY AXIS
############################

freqs = np.fft.fftshift(
    np.fft.fftfreq(nsamples, d=1/sample_rate)
)

############################
# HELPERS
############################

def capture_once(lo_freq):
    sdr.center_freq = lo_freq
    data = ugradio.sdr.capture_data(
        sdr,
        nsamples=nsamples,
        nblocks=nblocks
    )

    fft = np.fft.fftshift(np.fft.fft(data, axis=-1), axes=-1)
    power = np.abs(fft)**2
    return np.mean(power, axis=0)


def smooth(spec, width):
    return np.convolve(spec, np.ones(width)/width, mode='same')

############################
# LIVE LOOP
############################

lo_upper = HI_FREQ - freq_offset
lo_lower = HI_FREQ + freq_offset

avg_upper = np.zeros(nsamples)
avg_lower = np.zeros(nsamples)

count = 0

plt.ion()
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_xlabel("Frequency (MHz)")
ax.set_ylabel("Power (arb)")
ax.set_title("Live Hydrogen 21-cm Detection")

while True:

    spec_u = capture_once(lo_upper)
    spec_l = capture_once(lo_lower)

    avg_upper += spec_u
    avg_lower += spec_l
    count += 1

    if count % update_interval == 0:

        diff = (avg_upper - avg_lower) / count
        diff = smooth(diff, smooth_width)

        rf_freqs = freqs + lo_upper

        line.set_xdata(rf_freqs / 1e6)
        line.set_ydata(diff)

        ax.relim()
        ax.autoscale_view()
        plt.pause(0.01)

        print(f"Integrated {count} cycles")

