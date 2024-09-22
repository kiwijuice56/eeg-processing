import numpy as np
from scipy import signal
import yasa

import matplotlib.pyplot as plt


# Calculates bandpower of different frequencies in EEG data


raw_alpha_time = np.fromfile("data/raw_alpha_time.npy")
raw_alpha_value = np.fromfile("data/raw_alpha_value.npy")
raw_eeg_time = np.fromfile("data/raw_eeg_time.npy")
raw_eeg_value = np.fromfile("data/raw_eeg_value.npy")

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
# https://github.com/HPI-CH/UNIVERSE/blob/main/Features/features.py
# https://web.archive.org/web/20181105231756/http://developer.choosemuse.com/tools/available-data#Absolute_Band_Powers
# https://mind-monitor.com/forums/viewtopic.php?t=1651

window_size = 256
window_gap = 25

test_alpha_time = []
test_alpha_value = []

for i in range(window_size, len(raw_eeg_time), window_gap):
    window = raw_eeg_value[i - window_size : i]
    if len(window) < window_size:
        break
    freqs, psd = signal.welch(x=window, fs=256.0, window="hann")
    bandpower = yasa.bandpower_from_psd_ndarray(psd, freqs) # Calculate the bandpower on 3-D PSD array

    test_alpha_time.append(raw_eeg_time[i])
    test_alpha_value.append(bandpower[1])

plt.plot(raw_alpha_time, raw_alpha_value)
plt.plot(test_alpha_time, test_alpha_value)
plt.show()