import numpy as np
from scipy import signal
import yasa

import matplotlib.pyplot as plt

import json
import re


def interpolate_signal(x, y):
    min_time, max_time = x[0], x[-1]
    new_sample_count = len(y) # Replace with frequency calculations later

    new_x = np.linspace(min_time, max_time, num=new_sample_count)
    new_y = np.interp(new_x, x, y)

    # Convert to seconds and set start to 0
    new_x *= 10 ** -6
    new_x -= new_x[0]

    return new_x, new_y


# Open raw Muse data from JSON file, recorded using GD Muse
raw_data = {}
with open("data/test_eeg_and_psd5.json") as f:
    raw_json_data = str(f.readline())

    # Python's json module crashes if the file contains 'nan' rather than 'NaN'
    raw_json_data = re.sub(r'\bnan\b', 'NaN', raw_json_data)

    raw_data = json.loads(raw_json_data)


# Convert data into more usable form
raw_alpha_time = np.array(raw_data["alpha_absolute"]["time"])
raw_alpha_value = np.array(raw_data["alpha_absolute"]["value"])

raw_eeg_time = np.array(raw_data["eeg"]["time"])
raw_eeg_value = np.array(raw_data["eeg"]["value"])

# Interpolate data with consistent spacing between samples
# (Channel 1 is left forehead)
raw_eeg_time, raw_eeg_value = interpolate_signal(raw_eeg_time, raw_eeg_value[:, 1])
raw_alpha_time, raw_alpha_value = interpolate_signal(raw_alpha_time, raw_alpha_value[:, 1])

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

print(len(test_alpha_time))
print(len(raw_alpha_time))

plt.plot(raw_alpha_time, raw_alpha_value)
plt.plot(test_alpha_time, test_alpha_value)
plt.show()