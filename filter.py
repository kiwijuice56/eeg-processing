import numpy as np
import mne

import matplotlib.pyplot as plt

for trial in ["pink_noise_test_1_eeg", "binaural_theta_test_1_eeg"]:
    raw_eeg_time = np.fromfile("data/%s_time.npy" % trial)
    raw_eeg_value = np.fromfile("data/%s_value.npy" % trial)

    # Remove powerline interference and other hardware-specific noise (Banville et al)
    filtered_eeg_value = mne.filter.notch_filter(raw_eeg_value, Fs=256, freqs=[16, 21.3, 32, 42.7, 50, 60])

    # Save filtered signal
    filtered_eeg_value.tofile("data/filtered_%s_value.npy" % trial)

    plt.plot(raw_eeg_time[0:1024], raw_eeg_value[0:1024])
    plt.plot(raw_eeg_time[0:1024], filtered_eeg_value[0:1024])

plt.show()