import numpy as np
import scipy

import matplotlib.pyplot as plt


# Calculates bandpower of different frequencies in EEG data


# https://raphaelvallat.com/bandpower.html
def bandpower(data, sf, band, window_sec=None, relative=False):
    """Compute the average power of the signal x in a specific frequency band.

    Parameters
    ----------
    data : 1d-array
        Input signal in the time-domain.
    sf : float
        Sampling frequency of the data.
    band : list
        Lower and upper frequencies of the band of interest.
    window_sec : float
        Length of each window in seconds.
        If None, window_sec = (1 / min(band)) * 2
    relative : boolean
        If True, return the relative power (= divided by the total power of the signal).
        If False (default), return the absolute power.

    Return
    ------
    bp : float
        Absolute or relative band power.
    """
    from scipy.signal import welch
    from scipy.integrate import simpson
    band = np.asarray(band)
    low, high = band

    # Define window length
    if window_sec is not None:
        nperseg = window_sec * sf
    else:
        nperseg = (2 / low) * sf

    # Compute the modified periodogram (Welch)
    freqs, psd = welch(data, sf, nperseg=nperseg)

    # Frequency resolution
    freq_res = freqs[1] - freqs[0]

    # Find closest indices of band in frequency vector
    idx_band = np.logical_and(freqs >= low, freqs <= high)

    # Integral approximation of the spectrum using Simpson's rule.
    bp = simpson(psd[idx_band], dx=freq_res)

    bp = np.log(bp)

    if relative:
        bp /= simpson(psd, dx=freq_res)
    return bp


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

# This is slightly incorrect; the interpolated alpha signals start at t = 0,
# but they actually have a short delay before Muse sends any packets.
# In any case, this is still useful to see if the overall pattern is the same
plt.plot(raw_alpha_time, raw_alpha_value)
plt.plot(test_alpha_time, test_alpha_value)
plt.show()