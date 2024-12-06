import numpy as np
import mne

import matplotlib.pyplot as plt


# Filters each signal to remove noise and trends

def filter_trial():
    for trial in ["sleep_10_30"]:
        raw_eeg_time = np.fromfile("data/%s_eeg_time.npy" % trial)
        raw_eeg_value = np.fromfile("data/%s_eeg_value.npy" % trial)

        # Remove powerline interference and other hardware-specific noise (Banville et al)
        filtered_eeg_value = mne.filter.notch_filter(raw_eeg_value, Fs=256, freqs=[16, 21.3, 32, 42.7, 50, 60])

        # Remove constant DC component with a high pass filter
        filtered_eeg_value = mne.filter.filter_data(filtered_eeg_value, sfreq=256, l_freq=0.5, h_freq=None)

        # Save filtered signal
        filtered_eeg_value.tofile("data/filtered_%s_value.npy" % trial)

        # Smooth bandpower signals
        for bandpower in ["alpha", "beta", "gamma", "theta", "delta"]:
            raw_time = np.fromfile("data/%s_%s_time.npy" % (trial, bandpower))
            raw_value = np.fromfile("data/%s_%s_value.npy" % (trial, bandpower))

            # Remove high frequency noise using a hamming window
            smoothed_value = np.convolve(raw_value, np.hamming(512), mode="valid")

            # We have a new time range because we had to trim samples when smooothing
            new_time = np.linspace(raw_time[0], raw_time[-1], len(smoothed_value))

            new_time.tofile("data/smoothed_%s_%s_time.npy" % (trial, bandpower))
            smoothed_value.tofile("data/smoothed_%s_%s_value.npy" % (trial, bandpower))

        for channel in ["irh16", "ir", "red"]:
            raw_ppg_time = np.fromfile("data/%s_ppg_%s_time.npy" % (trial, channel))
            raw_ppg_value = np.fromfile("data/%s_ppg_%s_value.npy" % (trial, channel))

            filtered_ppg_value = mne.filter.filter_data(raw_ppg_value, sfreq=64, l_freq=0.5, h_freq=5.0)

            # Currently, we are not saving/using the ppg data for anything --
            # This plots an arbitrary section for demonstration purposes
            plt.plot(raw_ppg_time[100000:100000+256], filtered_ppg_value[100000:100000+256], label=channel)

    plt.title("ppg signals")
    plt.xlabel("time (s)")
    plt.legend()
    plt.show()

filter_trial()
