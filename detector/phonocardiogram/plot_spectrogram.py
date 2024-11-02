import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram

# 1. Load the data
with open("data.pickle", "rb") as f:
    signal, peak_locs = pickle.load(f)

# Ensure proper data format
signal = np.asarray(signal).flatten()
peak_locs = np.asarray(peak_locs).flatten()

# 2. Define and apply the bandpass filter
fs = 3000        # Sampling frequency in Hz
low_cutoff = 30  # Low cutoff frequency in Hz
high_cutoff = 70 # High cutoff frequency in Hz

def bandpass_filter(data, low_cutoff, high_cutoff, fs, order=2):
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype="bandpass")
    y = filtfilt(b, a, data)
    return y

filtered_signal = bandpass_filter(signal, low_cutoff, high_cutoff, fs)

# 3. Compute the spectrogram
nperseg = 1024        # Window size: 1,024 samples (~0.341 seconds)
noverlap = 512        # 50% overlap
window = 'hann'       # Hann window
nfft = nperseg        # FFT points

f, t_spec, Sxx = spectrogram(
    filtered_signal,
    fs=fs,
    window=window,
    nperseg=nperseg,
    noverlap=noverlap,
    nfft=nfft,
    scaling='density',
    mode='magnitude'
)

# 4. Convert peak locations to time
peak_times = peak_locs / fs

# 5. Plot the spectrogram and overlay peaks
plt.figure(figsize=(14, 6))
plt.pcolormesh(t_spec, f, Sxx, shading='gouraud', cmap='viridis')
plt.colorbar(label='Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram of Original Signal')
plt.ylim(0, 200)  # Cut off at 200 Hz

# Overlay peaks as vertical lines
for idx, peak_time in enumerate(peak_times):
    if idx == 0:
        plt.axvline(x=peak_time, color='red', linestyle='--', linewidth=1.5, label='Peak')
    else:
        plt.axvline(x=peak_time, color='red', linestyle='--', linewidth=1.5)

# Add legend
plt.legend(loc='upper right')

# Display the plot
plt.tight_layout()
plt.show()
