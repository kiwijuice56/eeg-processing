import numpy as np

import json
import re

import matplotlib.pyplot as plt


# Converts raw JSON data from the Muse recorder into a binary format, normalized
# to seconds instead of microseconds


def interpolate_signal(x, y, old_frequency=-1, new_frequency=-1):
    new_sample_count = int(new_frequency / old_frequency * len(y))

    if old_frequency < 0:
        new_sample_count = len(y)

    # The timestamps themselves are worthless (except for the end points)...
    # They only reflect the time the Bluetooth packet was sent/received
    old_x = np.linspace(x[0], x[-1], num=len(y))
    new_x = np.linspace(x[0], x[-1], num=new_sample_count)
    new_y = np.interp(new_x, old_x, y)

    # Convert to seconds and set start to 0
    new_x -= x[0]
    new_x *= 10 ** -6

    return new_x, new_y


def eeg_from_json_to_npy(file_name, new_file_name, signal_name, channels=(1,), old_frequency=-1, new_frequency=-1, plot=False):
    # Open raw Muse data from JSON file, recorded using GD Muse
    raw_data = {}
    with open(file_name) as f:
        raw_json_data = str(f.readline())

        # Python's json module crashes if the file contains 'nan' rather than 'NaN'
        # Note that 'nan' values represent dropped packets that should be linearly
        # interpolated! For now, this script assumes the connection is stable
        raw_json_data = re.sub(r'\bnan\b', 'NaN', raw_json_data)

        raw_data = json.loads(raw_json_data)

    # Timestamps of Bluetooth packets ...
    # Not very useful except for the start and end
    raw_time = np.array(raw_data[signal_name]["time"])

    # Average over the selected EEG channels
    avg_value = []
    for channel in channels:
        avg_value.append(np.array(raw_data[signal_name]["value"])[:, channel])
    avg_value = np.average(avg_value, axis=0)

    # Interpolate data with consistent spacing between samples
    interp_raw_time, interp_raw_value = interpolate_signal(raw_time, avg_value, old_frequency=old_frequency, new_frequency=new_frequency)

    # Save interpolated signals
    interp_raw_time.tofile(new_file_name % "time")
    interp_raw_value.tofile(new_file_name % "value")

    if plot:
        plt.plot(interp_raw_time, interp_raw_value, label=file_name)


for trial in ["pink_noise_test_1", "binaural_theta_test_1"]:
    source = "data/%s.json" % trial
    channels = (0,) # Left ear

    eeg_from_json_to_npy(source, "data/" + trial + "_alpha_%s.npy", "alpha_absolute", channels, plot=False)
    eeg_from_json_to_npy(source, "data/" + trial + "_beta_%s.npy", "beta_absolute", channels, plot=False)
    eeg_from_json_to_npy(source, "data/" + trial + "_gamma_%s.npy", "gamma_absolute", channels, plot=True)
    eeg_from_json_to_npy(source, "data/" + trial + "_theta_%s.npy", "theta_absolute", channels, plot=False)
    eeg_from_json_to_npy(source, "data/" + trial + "_delta_%s.npy", "delta_absolute", channels, plot=False)

    eeg_from_json_to_npy(source, "data/" + trial + "_eeg_%s.npy", "eeg", channels, plot=False)

plt.legend()
plt.show()