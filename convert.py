import numpy as np

import matplotlib.pyplot as plt

import json
import re


# Converts raw JSON data from the Muse recorder into a binary format, normalized
# to seconds instead of microseconds


def interpolate_signal(x, y):
    min_time, max_time = x[0], x[-1]
    new_sample_count = 4*len(y) # Replace with frequency calculations later

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

raw_averaged_eeg_value = []
raw_averaged_eeg_time = []

# MuseS samples in "bursts," measuring the EEG 12 times within ~5000 microseconds intervals,
# then waiting a long period to repeat. Here, we take the average of the samples for one clean signal.
maximum_sample_wait = 5000
minimum_time = raw_eeg_time[0]
i = 0
while i < len(raw_eeg_value):
    sample_value = []
    sample_time = []
    j = i
    while j < len(raw_eeg_value):
        if raw_eeg_time[j] - raw_eeg_time[i] < maximum_sample_wait:
            # (Channel 1 is left forehead)
            sample_value.append(raw_eeg_value[j][1])
            sample_time.append(raw_eeg_time[j])
        else:
            break
        j = j + 1
    i = j
    raw_averaged_eeg_value.append(sum(sample_value) / len(sample_value))
    # Although we already normalize the time later, subtracting the minimum time here prevents integer overflows
    raw_averaged_eeg_time.append(sum([time - minimum_time for time in sample_time]) // len(sample_time))

raw_averaged_eeg_value = np.array(raw_averaged_eeg_value)
raw_averaged_eeg_time = np.array(raw_averaged_eeg_time)

# Interpolate data with consistent spacing between samples
raw_averaged_eeg_time, raw_averaged_eeg_value = interpolate_signal(raw_averaged_eeg_time, raw_averaged_eeg_value)
raw_alpha_time, raw_alpha_value = interpolate_signal(raw_alpha_time, raw_alpha_value[:, 1])

# Center the signals around y = 0
# raw_averaged_eeg_value -= np.median(raw_averaged_eeg_value)
# raw_alpha_value -= np.median(raw_alpha_value)

plt.plot(raw_averaged_eeg_time, raw_averaged_eeg_value)

# # Save interpolated signals
raw_alpha_time.tofile("data/raw_alpha_time.npy")
raw_alpha_value.tofile("data/raw_alpha_value.npy")
raw_averaged_eeg_time.tofile("data/raw_eeg_time.npy")
raw_averaged_eeg_value.tofile("data/raw_eeg_value.npy")

plt.show()