import numpy as np

import matplotlib.pyplot as plt

# Calculates bandpower of different frequencies in EEG data

# Experimental, based on sparse Muse documentation:
# Calculates the bandpower of a window of the EEG signal
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.welch.html
# https://github.com/HPI-CH/UNIVERSE/blob/main/Features/features.py
# https://web.archive.org/web/20181105231756/http://developer.choosemuse.com/tools/available-data#Absolute_Band_Powers
# https://mind-monitor.com/forums/viewtopic.php?t=1651
def calculate_bandpower(data, sampling_frequency, band):
    fft = np.fft.fft(data)
    psd = np.square(np.abs(fft))
    bandpower = 0
    for i in range(len(psd)):
        freq = i * sampling_frequency / len(psd)
        if band[0] <= freq <= band[1]:
            bandpower += psd[i]
    return np.log(bandpower)


# Uses a window to calculate the bandpower of an EEG signal over time
def calculate_bandpower_signal(eeg_time, eeg_value, band, sampling_frequency, window_size=2048, window_gap=256, plot=True):
    signal_time = []
    signal_value = []

    for i in range(window_size, len(raw_eeg_time), window_gap):
        window_value = eeg_value[i - window_size : i]

        signal_time.append(eeg_time[i])
        signal_value.append(calculate_bandpower(window_value, sampling_frequency, band))

    # Normalize to between 0 and 1
    signal_value = (signal_value - min(signal_value)) / (max(signal_value) - min(signal_value))

    if plot:
        plt.plot(signal_time, signal_value)

    return np.array(signal_time), np.array(signal_value)


raw_alpha_time = np.fromfile("data/raw_alpha_time.npy")
raw_alpha_value = np.fromfile("data/raw_alpha_value.npy")
raw_eeg_time = np.fromfile("data/raw_eeg_time.npy")
filtered_eeg_value = np.fromfile("data/filtered_eeg_value.npy")

# Muse's built-in values
plt.plot(raw_alpha_time, raw_alpha_value)

# calculate_bandpower_signal(raw_eeg_time, filtered_eeg_value, [1.0, 4.0], 1024.0) # Delta
# calculate_bandpower_signal(raw_eeg_time, filtered_eeg_value, [4.0, 8.0], 1024.0) # Theta
calculate_bandpower_signal(raw_eeg_time, filtered_eeg_value, [8.0, 13.0], 1024.0) # Alpha
# calculate_bandpower_signal(raw_eeg_time, filtered_eeg_value, [13.0, 30.0], 1024.0) # Beta
# calculate_bandpower_signal(raw_eeg_time, filtered_eeg_value, [30.0, 44.0], 1024.0) # Gamma



plt.show()