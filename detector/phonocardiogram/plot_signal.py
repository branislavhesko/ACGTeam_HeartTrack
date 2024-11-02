import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram

# Load the data
with open("data.pickle", "rb") as f:
    signal, peak_locs = pickle.load(f)

# Ensure the signal is a 1D array
signal = np.asarray(signal).flatten()

fs = 3000  # Sampling frequency in Hz
low_cutoff = 30  # Low cutoff frequency in Hz
high_cutoff = 70  # High cutoff frequency in Hz

def bandpass_filter(data, low_cutoff, high_cutoff, fs, order=2):
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype="bandpass")
    y = filtfilt(b, a, data)
    return y

# Apply bandpass filter
filtered_signal = bandpass_filter(signal, low_cutoff, high_cutoff, fs)

# Plot the original and filtered signals
plt.figure(figsize=(14, 6))
plt.plot(signal[20000:23000], label="Original Signal", alpha=0.7)
plt.plot(filtered_signal[20000:23000], label="Filtered Signal", alpha=0.7)
for idx, loc in enumerate(peak_locs[7:9]):
    plt.axvline(x=loc - 20000, color="red", linestyle="--", label="Peak" if idx == 0 else "")
plt.xlabel("Sample Index")
plt.ylabel("Amplitude")
plt.title("Original and Filtered Signal")
plt.legend()
plt.tight_layout()
plt.show()

# Compute the spectrogram using scipy.signal.spectrogram
f, t, Sxx = spectrogram(signal, fs=fs, nperseg=1024 * 2, noverlap=512 * 2, scaling='density', mode='magnitude')

# Plot the spectrogram
plt.figure(figsize=(14, 6))
plt.pcolormesh(t, f, Sxx, shading='gouraud', cmap='viridis')
plt.colorbar(label='Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.title('Spectrogram of Filtered Signal')
plt.ylim(0, fs / 2)  # Display up to Nyquist frequency
plt.tight_layout()
plt.show()