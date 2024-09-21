import numpy as np
import scipy

import matplotlib.pyplot as plt

import json
import re

raw_data = {}

with open("data/test_eeg_and_psd.json") as f:
    raw_json_data = str(f.readline())

    # Python's json module crashes if the file contains 'nan' rather than 'NaN'
    raw_json_data = re.sub(r'\bnan\b', 'NaN', raw_json_data)

    raw_data = json.loads(raw_json_data)


raw_alpha_time = np.array(raw_data["alpha_absolute"]["time"])
raw_alpha_value = np.transpose(np.array(raw_data["alpha_absolute"]["value"]))

plt.plot(raw_alpha_time, raw_alpha_value[1]) # left_forehead
plt.plot(raw_alpha_time, raw_alpha_value[2]) # right_forehead
plt.show()