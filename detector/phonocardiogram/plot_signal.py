import pickle

import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

with open("data.pickle", "rb") as f:
    signal, peak_locs = pickle.load(f)

fs = 3000
low_cutoff = 30
high_cutoff = 70

def bandpass_filter(data, low_cutoff, high_cutoff, fs, order=2):
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype="bandpass")
    y = filtfilt(b, a, data)
    return y

filtered_signal = bandpass_filter(signal.flatten(), low_cutoff, high_cutoff, fs)
plt.plot(signal[20000:23000])
plt.plot(filtered_signal[20000:23000])


for loc in peak_locs[7:9]:
    plt.axvline(x=loc - 20000, color="red", linestyle="--", label="Peak" if loc == peak_locs[0] else "")

plt.show()
