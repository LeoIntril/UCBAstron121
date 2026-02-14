import numpy as np
import ugradio
import os
import matplotlib.pyplot as plt
import asyncio

############################
# CONFIGURATION
############################

HI_FREQ = 1420.405751e6
c = 299792458

sample_rate = 1e6
nsamples = 2**14
nblocks = 8
gain = 40
freq_offset = 100e3

n_cal_integrations = 50
n_obs_integrations = 300
live_update_interval = 5

T_hot = 300
T_cold = 10

outdir = "data"
os.makedirs(outdir, exist_ok=True)

############################
# HELPER FUNCTIONS
############################

async def capture_samples(sdr, num_buffers=1):
    """Async capture returning concatenated array of samples"""
    samples = []
    async for buf in sdr.stream():
        samples.append(buf)
        if len(samples) >= num_buffers:
            await sdr.stop()
            break
    return np.concatenate(samples, axis=0)

async def integrate(lo_freq, n_integrations, label):
    """Integrate spectrum with live plot for a given LO frequency"""
    avg_spec = np.zeros(nsamples)

    plt.ion()
    fig, ax = plt.subplots()
    freqs = np.fft.fftshift(np.fft.fftfreq(nsamples, d=1/sample_rate))
    rf_freqs = freqs + lo_freq
    line, = ax.plot(rf_freqs/1e6, avg_spec)
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("Power")
    ax.set_title(label)

    for i in range(n_integrations):
        # Capture new SDR object for each integration to avoid frozen stream
        sdr = ugradio.sdr.RtlSdr()
        sdr.sample_rate = sample_rate
        sdr.gain = gain
        sdr.direct = False
        sdr.center_freq = lo_freq

        data = await capture_samples(sdr, 1)
        sdr.close()  # close immediately

        if len(data) < nsamples:
            continue

        fft = np.fft.fftshift(np.fft.fft(data[:nsamples]))
        power = np.abs(fft)**2
        avg_spec += power

        if (i+1) % live_update_interval == 0:
            line.set_ydata(avg_spec / (i+1))
            ax.relim()
            ax.autoscale_view()
            plt.draw()
            plt.pause(0.01)
            print(f"{label}: {i+1}/{n_integrations}")

    plt.ioff()
    plt.show()
    return avg_spec / n_integrations

############################
# MAIN FLOW
############################

async def main():
    try:
        print("\n=== INTENSITY CALIBRATION ===")
        await asyncio.to_thread(input, "Aim horn at COLD SKY and press Enter…")
        s_cold = await integrate(HI_FREQ, n_cal_integrations, "Cold Sky")

        await asyncio.to_thread(input, "Place HOT Load in front of horn and press Enter…")
        s_hot = await integrate(HI_FREQ, n_cal_integrations, "Hot Load")

        P_cold = np.mean(s_cold)
        P_hot = np.mean(s_hot)
        Y = P_hot / P_cold
        T_sys = (T_hot - Y*T_cold) / (Y - 1)
        print(f"\nY-factor: {Y:.4f}, System temperature: {T_sys:.2f} K")

        await asyncio.to_thread(input, "\nAim horn at target and press Enter…")

        lo_upper = HI_FREQ - freq_offset
        lo_lower = HI_FREQ + freq_offset

        spec_upper = await integrate(lo_upper, n_obs_integrations, "HI Upper LO")
        spec_lower = await integrate(lo_lower, n_obs_integrations, "HI Lower LO")

        diff_spec = spec_upper - spec_lower
        T_ant = T_sys * (diff_spec / P_cold)

        freqs = np.fft.fftshift(np.fft.fftfreq(nsamples, d=1/sample_rate))
        rf_freqs = freqs + lo_upper
        velocity = c * (HI_FREQ - rf_freqs) / HI_FREQ / 1000

        unix_time = ugradio.timing.unix_time()
        lat = ugradio.nch.lat
        lon = ugradio.nch.lon

        np.savez(
            f"{outdir}/hi_ugradio_reinit.npz",
            freq_Hz=rf_freqs,
            velocity_kms=velocity,
            temperature_K=T_ant,
            s_cold=s_cold,
            s_hot=s_hot,
            spec_upper=spec_upper,
            spec_lower=spec_lower,
            diff_spec=diff_spec,
            T_sys_K=T_sys,
            Y_factor=Y,
            HI_rest_freq_Hz=HI_FREQ,
            lo_upper_Hz=lo_upper,
            lo_lower_Hz=lo_lower,
            sample_rate_Hz=sample_rate,
            gain_dB=gain,
            nsamples=nsamples,
            nblocks=nblocks,
            n_cal_integrations=n_cal_integrations,
            n_obs_integrations=n_obs_integrations,
            unix_time=unix_time,
            latitude_deg=lat,
            longitude_deg=lon
        )

        print("\nObservation complete.")
        print(f"Saved to {outdir}/hi_ugradio_reinit.npz")

    except KeyboardInterrupt:
        print("\nUser interrupted!")

############################
# RUN
############################

asyncio.run(main())
