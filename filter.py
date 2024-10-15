import numpy as np
import mne

import matplotlib.pyplot as plt

raw_eeg_time = np.fromfile("data/eeg_2_time.npy")
raw_eeg_value = np.fromfile("data/eeg_2_value.npy")

# Remove powerline interference and other hardware-specific noise (Banville et al)
filtered_eeg_value = mne.filter.notch_filter(raw_eeg_value, Fs=1024, freqs=[16, 21.3, 32, 42.7, 50, 60])

# Save filtered signal
filtered_eeg_value.tofile("data/filtered_eeg_2_value.npy")

plt.plot(raw_eeg_time[0:1024], raw_eeg_value[0:1024])
plt.plot(raw_eeg_time[0:1024], filtered_eeg_value[0:1024])
plt.show()