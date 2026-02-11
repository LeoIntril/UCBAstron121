def capture_spectrum(
    sdr,
    lo_freq,
    label="",
    ntrials=4,
    outdir="data"
):
    sdr.center_freq = lo_freq
    print(f"\nTuned LO to {lo_freq/1e6:.6f} MHz ({label})")

    spectra = []

    for trial in range(ntrials):
        print(f"  Trial {trial+1}/{ntrials}")

        # Synchronous read — NO async objects
        data = sdr.read_samples(nsamples * nblocks)

        # Reshape into blocks
        data = data.reshape((nblocks, nsamples))

        # FFT along time axis
        fft = np.fft.fftshift(
            np.fft.fft(data, axis=1),
            axes=1
        )

        power = np.abs(fft)**2
        spec = np.mean(power, axis=0)
        spectra.append(spec)

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
