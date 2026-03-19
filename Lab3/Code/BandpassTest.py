import snap_spec.snap as snap
import matplotlib.pyplot as plt
import numpy as np

spec = snap.UGRadioSnap()
spec.initialize(mode='corr')

data = spec.read_data()
vis = data["corr01"]

plt.plot(np.abs(vis))
plt.xlabel("Channel number (0–1023)")
plt.ylabel("Amplitude")
plt.title("Raw spectrum — use this to identify band edges")
plt.show()
