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


# Loads a JSON file and returns the data it contains
def from_json_to_py(file_name):
    # Open raw Muse data from JSON file, recorded using GD Muse
    raw_data = []
    with open(file_name) as f:
        raw_json_data = str(f.readline())

        # Python's json module crashes if the file contains 'nan' rather than 'NaN'
        # Note that 'nan' values represent dropped packets that should be linearly
        # interpolated! For now, this script assumes the connection is stable
        raw_json_data = re.sub(r'\bnan\b', 'NaN', raw_json_data)
        raw_data = json.loads(raw_json_data)
    return raw_data


# Takes a signal with multiple channels (EEG, PPG), averages the given channels, resamples it, then converts it to a .npy file
def process_signal(raw_data, new_file_name, signal_name, channels=(1,), old_frequency=-1, new_frequency=-1, plot=False):
    # Timestamps of Bluetooth packets ...
    # Not very useful except for the start and end
    raw_time = np.array(raw_data[signal_name]["time"])

    # Average over the selected channels
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
        plt.plot(interp_raw_time, interp_raw_value, label=signal_name)


# Convert and resample Muse recordings
def convert_trial_data():
    eeg_channels = (0,) # 0 corresponds to left ear electrode... See Muse SDK documentation for different channel mappings

    for trial in ["sleep_10_30"]:
        source = "data/%s.json" % trial
        raw_data = from_json_to_py(source)

        process_signal(raw_data, "data/" + trial + "_alpha_%s.npy", "alpha_absolute", eeg_channels, plot=True)
        process_signal(raw_data, "data/" + trial + "_beta_%s.npy", "beta_absolute", eeg_channels, plot=False)
        process_signal(raw_data, "data/" + trial + "_gamma_%s.npy", "gamma_absolute", eeg_channels, plot=False)
        process_signal(raw_data, "data/" + trial + "_theta_%s.npy", "theta_absolute", eeg_channels, plot=False)
        process_signal(raw_data, "data/" + trial + "_delta_%s.npy", "delta_absolute", eeg_channels, plot=False)

        process_signal(raw_data, "data/" + trial + "_eeg_%s.npy", "eeg", eeg_channels, plot=False)

        # Convert each channel of the PPG signal individually
        process_signal(raw_data, "data/" + trial + "_ppg_irh16_%s.npy", "ppg", (0,), plot=False)
        process_signal(raw_data, "data/" + trial + "_ppg_ir_%s.npy", "ppg", (1,), plot=False)
        process_signal(raw_data, "data/" + trial + "_ppg_red_%s.npy", "ppg", (2,), plot=False)

    plt.xlabel("time (s)")
    plt.title("alpha signal")
    plt.legend()
    plt.show()

convert_trial_data()