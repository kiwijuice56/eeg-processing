import numpy as np

import matplotlib.pyplot as plt


# Calculates bandpower of different frequencies in EEG data


def get_psd_window(data, sampling_frequency, band):
    fft = np.fft.fft(data)
    psd = np.square(np.abs(fft))
    psd_window_freq = []
    psd_window_val = []

    for i in range(len(fft)):
        freq = i * sampling_frequency / len(fft)
        if band[0] <= freq <= band[1]:
            psd_window_freq.append(freq)
            psd_window_val.append(np.abs(psd[i]))

    return psd_window_freq, psd_window_val


def get_fft_window(data, sampling_frequency, band):
    fft = np.fft.fft(data)
    fft_window_freq = []
    fft_window_val = []

    for i in range(len(fft)):
        freq = i * sampling_frequency / len(fft)
        if band[0] <= freq <= band[1]:
            fft_window_freq.append(freq)
            fft_window_val.append(np.abs(fft[i]))

    return fft_window_freq, fft_window_val


# Experimental, based on sparse Muse documentation:
# Calculates the bandpower of a window of the EEG signal
def calculate_bandpower(data, sampling_frequency, band, plot=False):
    fft = np.fft.fft(data)
    psd = np.square(np.abs(fft))
    bandpower = 0

    for i in range(len(psd)):
        freq = i * sampling_frequency / len(psd)
        if band[0] <= freq <= band[1]:
            bandpower += psd[i]

    if plot:
        plt.plot(fft)

    return np.log(bandpower)


# Uses a window to calculate the bandpower of an EEG signal over time
def calculate_bandpower_signal(eeg_time, eeg_value, band, sampling_frequency, window_size=2048, window_gap=512, plot=True):
    signal_time = []
    signal_value = []

    for i in range(window_size, len(eeg_time), window_gap):
        window_value = eeg_value[i - window_size : i]

        signal_time.append(eeg_time[i])
        signal_value.append(calculate_bandpower(window_value, sampling_frequency, band))

    # Normalize to between 0 and 1
    signal_value = (signal_value - min(signal_value)) / (max(signal_value) - min(signal_value))

    if plot:
        plt.plot(signal_time, signal_value)

    return np.array(signal_time), np.array(signal_value)


def test_a():
    raw_muse_time = np.fromfile("data/beta_2_time.npy")
    raw_muse_value = np.fromfile("data/beta_2_value.npy")
    raw_eeg_time = np.fromfile("data/eeg_2_time.npy")
    filtered_eeg_value = np.fromfile("data/filtered_eeg_2_value.npy")

    # Normalize built-in values
    raw_muse_value -= min(raw_muse_value)
    raw_muse_value /= max(raw_muse_value)

    plt.plot(raw_muse_time, raw_muse_value)

    # calculate_bandpower_signal(raw_eeg_time, filtered_eeg_value, [7.5, 13.0], 1024.0)     # Alpha
    calculate_bandpower_signal(raw_eeg_time, filtered_eeg_value, [13.0, 30.0], 1024.0)      # Beta
    # calculate_bandpower_signal(raw_eeg_time, filtered_eeg_value, [1.0, 4.0], 1024.0)      # Delta
    # calculate_bandpower_signal(raw_eeg_time, filtered_eeg_value, [4.0, 8.0], 1024.0)      # Theta
    # calculate_bandpower_signal(raw_eeg_time, filtered_eeg_value, [30.0, 44.0], 1024.0)    # Gamma

    plt.show()


def test_b():
    filtered_eeg_value = np.fromfile("data/filtered_binaural_theta_test_1_eeg_value.npy")

    psd_window_freq, psd_window_val = get_psd_window(filtered_eeg_value, 1024.0, band=[0.1, 50.0])

    plt.plot(psd_window_freq, psd_window_val, label="binaural_theta_fft")

    plt.legend()
    plt.show()


def test_c():
    filtered_eeg_value = np.fromfile("data/filtered_pink_noise_test_1_eeg_value.npy")
    fft_window_freq, fft_window_val = get_psd_window(filtered_eeg_value, 1024.0, band=[2, 10])
    plt.plot(fft_window_freq, fft_window_val, label="pink_noise_psd")

    filtered_eeg_value = np.fromfile("data/filtered_binaural_theta_test_1_eeg_value.npy")
    fft_window_freq, fft_window_val = get_psd_window(filtered_eeg_value, 1024.0, band=[2, 10])
    plt.plot(fft_window_freq, fft_window_val, label="binaural_theta_psd")

    plt.legend()
    plt.show()


test_c()
