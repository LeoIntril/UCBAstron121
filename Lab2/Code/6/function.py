import os
import numpy as np

def capture_spectrum(
    sdr,
    lo_freq,
    label="",
    ntrials=4,
    outdir="data"
):
    """
    Capture multiple power spectra at a given LO frequency using RTL-SDR.

    - Uses synchronous reads (read_samples)
    - Reshapes into (nblocks, nsamples)
    - Saves one NPZ per trial
    - Returns all spectra and their average
    """

    # Ensure output directory exists
    os.makedirs(outdir, exist_ok=True)

    # Tune LO
    sdr.center_freq = lo_freq
    print(f"\nTuned LO to {lo_freq/1e6:.6f} MHz ({label})")

    spectra = []

    for trial in range(ntrials):
        print(f"  Trial {trial+1}/{ntrials}")

        # Blocking read: returns 1D complex array
        data = sdr.read_samples(nsamples * nblocks)

        # Sanity check
        if data.size != nsamples * nblocks:
            raise RuntimeError("Incorrect number of samples returned from SDR")

        # Reshape into blocks
        data = data.reshape((nblocks, nsamples))

        # FFT along time axis
        fft = np.fft.fftshift(
            np.fft.fft(data, axis=1),
            axes=1
        )

        power = np.abs(fft)**2

        # Average over blocks
        spec = np.mean(power, axis=0)
        spectra.append(spec)

        # Filename
        fname = (
            f"{outdir}/"
            f"{label}_"
            f"{lo_freq/1e6:.3f}MHz_"
            f"trial{trial:02d}.npz"
        )

        # Save trial
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
