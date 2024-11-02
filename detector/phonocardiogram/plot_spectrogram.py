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

# 5. Create a figure with two subplots: spectrogram and original signal
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})

# -------------------------
# Top Subplot: Spectrogram
# -------------------------

# Plot the spectrogram on ax1
pcm = ax1.pcolormesh(t_spec, f, Sxx, shading='gouraud', cmap='viridis')
#fig.colorbar(pcm, ax=ax1, label='Magnitude')
ax1.set_ylabel('Frequency [Hz]')
ax1.set_title('Spectrogram of Filtered PCG Signal at fs = 3,000 Hz (Up to 200 Hz)')
ax1.set_ylim(0, 100)  # Limit frequency to 200 Hz

# Overlay peaks as vertical dashed red lines on spectrogram
for idx, peak_time in enumerate(peak_times):
    if idx == 0:
        ax1.axvline(x=peak_time, color='red', linestyle='--', linewidth=1.5, label='Peak')
    else:
        ax1.axvline(x=peak_time, color='red', linestyle='--', linewidth=1.5)

# Add legend for peaks
ax1.legend(loc='upper right')

# -------------------------------
# Bottom Subplot: Original Signal
# -------------------------------


time = np.arange(len(signal)) / fs  # Time in seconds

# Plot the original signal on ax2
ax2.plot(time, signal, label='Original Signal', color='blue', alpha=0.7)
ax2.set_xlabel('Time [sec]')
ax2.set_ylabel('Amplitude')
ax2.set_title('Original PCG Signal with Peaks')
ax2.set_xlim(t_spec.min(), t_spec.max())  # Align with spectrogram's time axis

# Annotate peaks on the original signal plot with vertical dashed red lines
for peak_time in peak_times:
    ax2.axvline(x=peak_time, color='red', linestyle='--', linewidth=1.0)

# Optionally, mark peaks with markers (uncomment if desired)
# ax2.plot(peak_times, signal[peak_locs], 'ro', label='Peak')

# Add legend for the signal plot
ax2.legend(loc='upper right')

# Enhance layout and display the plot
plt.tight_layout()
plt.show()
