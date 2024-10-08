import numpy as np
import mne

import matplotlib.pyplot as plt

raw_eeg_time = np.fromfile("data/eeg_2_time.npy")
raw_eeg_value = np.fromfile("data/eeg_2_value.npy")

# Normalize the data
raw_eeg_value -= np.median(raw_eeg_value)
raw_eeg_value /= np.std(raw_eeg_value)

# Remove powerline interference (Anders)
filtered_eeg_value = mne.filter.filter_data(raw_eeg_value, 1024, 0.5, 50)

# Save filtered signal
filtered_eeg_value.tofile("data/filtered_eeg_2_value.npy")

plt.plot(raw_eeg_time[0:1024], raw_eeg_value[0:1024])
plt.plot(raw_eeg_time[0:1024], filtered_eeg_value[0:1024])
plt.show()