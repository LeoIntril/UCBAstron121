import ugradio
from ugradio import dft, sdr
import numpy as np
import time



rates = [1.0e6, 2.0e6, 3.2e6]
for r in rates:
    r =int(r)
    filename= f"FreqRes{r}Hz, 16384"
N = 16384


    
#def capture_sine_wave(sample_rate, filename , use_custom_fir=False):
sample_rate = 1.0e6
filename = f"FreqRes{sample_rate}Hz, 16384"
sdr = ugradio.sdr.SDR(direct=True, sample_rate=sample_rate)


# disabling anti-aliasing FIR filter
fir_coeffs = np.array([0, 0, 0, 0, 0, 0, 0, 0,
                       0, 0, 0, 0, 0, 0, 0, 2047],
                      dtype=np.int16)
sdr.set_fir_coeffs(fir_coeffs)


_ = sdr.capture_data(N)


# Actual data capture
data = sdr.capture_data(nblocks=2)[1]

np.savez(filename, data=data, sample_rate=sample_rate, timestamp=time.time())
print(f"Data saved to {filename}")


sdr.close()

del(sdr)


