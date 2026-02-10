import numpy as np
import ugradio.sdr

# -----------------------------
# User-controlled parameters
# -----------------------------
fs = 2_000_000        # sampling rate (Hz)
N_block = 4096        # samples per block
N_blocks = 16         # allows 1,2,4,8,16 averaging

# -----------------------------
# SDR setup
# -----------------------------
sdr = ugradio.sdr.SDR(sample_rate=fs)

# Flush stale buffer (lab explicitly warns about this)
_ = sdr.read_samples(N_block)

# -----------------------------
# Acquire blocks
# -----------------------------
blocks = np.empty((N_blocks, N_block), dtype=float)

for i in range(N_blocks):
    x = sdr.read_samples(N_block)
    blocks[i] = x.astype(float)

# -----------------------------
# Save to disk
# -----------------------------
np.savez(
    "noise_blocks_fs2MHz.npz",
    blocks=blocks,        # shape (N_blocks, N_block)
    sample_rate=fs
)

print("Saved blocks with shape:", blocks.shape)
