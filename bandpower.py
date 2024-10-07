import numpy as np

import matplotlib.pyplot as plt

# Calculates bandpower of different frequencies in EEG data

# Experimental, based on sparse Muse documentation
def bandpower(data, sampling_freq, band):
    fft = np.fft.fft(data,)
    psd = np.square(np.abs(fft))
    bandpower = 0
    for i in range(len(psd)):
        freq = i * sampling_freq / len(psd)
        if band[0] <= freq <= band[1]:
            bandpower += psd[i]
    return np.log(bandpower)

raw_alpha_time = np.fromfile("data/raw_alpha_time.npy")
raw_alpha_value = np.fromfile("data/raw_alpha_value.npy")
raw_eeg_time = np.fromfile("data/raw_eeg_time.npy")
raw_eeg_value = np.fromfile("data/raw_eeg_value.npy")

# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
# https://github.com/HPI-CH/UNIVERSE/blob/main/Features/features.py
# https://web.archive.org/web/20181105231756/http://developer.choosemuse.com/tools/available-data#Absolute_Band_Powers
# https://mind-monitor.com/forums/viewtopic.php?t=1651

window_size = 256
window_gap = 26

test_alpha_time = []
test_alpha_value = []

for i in range(window_size, len(raw_eeg_time), window_gap):
    window_value = raw_eeg_value[i - window_size : i]

    test_alpha_time.append(raw_eeg_time[i])
    test_alpha_value.append(bandpower(window_value, 256.0, [7.5, 13.0]))

test_alpha_value = (test_alpha_value - min(test_alpha_value)) / (max(test_alpha_value) - min(test_alpha_value))

# This is slightly incorrect; the interpolated alpha signals start at t = 0,
# but they actually have a short delay before Muse sends any packets.
# In any case, this is still useful to see if the overall pattern is the same
plt.plot(raw_alpha_time, raw_alpha_value)
plt.plot(test_alpha_time, test_alpha_value)
plt.show()