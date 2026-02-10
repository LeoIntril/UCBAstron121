import ugradio
from ugradio import dft, sdr
import numpy as np
import time



rates = [1.0e6, 2.0e6, 3.2e6]
for r in rates:
    r =int(r)
    filename= f"testng_{r}Hz"
N = 18232


    
#def capture_sine_wave(sample_rate, filename , use_custom_fir=False):
sample_rate = 3.0e6
filename = f"testing_{sample_rate}Hz"
sdr = ugradio.sdr.SDR(direct=True, sample_rate=sample_rate)


#if use_custom_fir:
    #fir_coeffs = np.array()
    #sdr.set_fir_coeffs(fir_coeffs)



# Actual data capture
data = sdr.capture_data(nblocks=2)[1]

np.savez(filename, data=data, sample_rate=sample_rate, timestamp=time.time())
print(f"Data saved to {filename}")


sdr.close()

del(sdr)


