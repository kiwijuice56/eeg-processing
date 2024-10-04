import numpy as np
import mne

import matplotlib.pyplot as plt

raw_eeg_time = np.fromfile("data/raw_eeg_time.npy")
raw_eeg_value = np.fromfile("data/raw_eeg_value.npy")

# Remove powerline interference (Anders)
filtered_eeg_value = mne.filter.filter_data(raw_eeg_value, 256, 0.5, 50)

plt.plot(raw_eeg_time[0:512], raw_eeg_value[0:512])
plt.plot(raw_eeg_time[0:512], filtered_eeg_value[0:512])
plt.show()