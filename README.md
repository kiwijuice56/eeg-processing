# eeg-processing
A small set of scripts to process EEG data directly from a Muse headband into useful brain signals. Please note that these scripts are incomplete -- they require some tinkering to work with different signals. Please reach out to me at ealfaro@mit.edu if you have any questions! This code was created by Eric Alfaro as part of a Fall UROP 2024 with Dr. Rich Fletcher.

## Structure
- `convert.py`: Contains several helper methods to convert Muse data from a `.json` format (as given by the GDMuse recording program) into a `.npy` format. Also oversamples signals.
- `filter.py`: Filters signals from `convert.py` to denoise and detrend.
- `bandpower.py`: Contains several test procedures for analyzing filtered EEG data, such as plotting the PSD or calculating the bandpower signals.
