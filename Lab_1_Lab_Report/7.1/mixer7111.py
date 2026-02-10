import ugradio
from ugradio import sdr
import numpy as np
import time

fs = 1_000_000
N  = 16384

sdr = ugradio.sdr.SDR(sample_rate=fs)

# flush stale buffer
_ = sdr.read_samples(4096)

# capture IF
x = sdr.read_samples(N).astype(float)

np.savez("dsb_IF_RF9p8_LO10_fs1MHz.npz", data=x, sample_rate=fs)

print("saved", x.shape)
