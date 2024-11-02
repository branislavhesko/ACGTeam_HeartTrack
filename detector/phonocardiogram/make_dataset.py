import pickle

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, spectrogram


def yield_raw_data():
    mat = loadmat('PCG_dataset.mat')
    pcg = mat['PCG_dataset']

    for example_idx in range(len(pcg[0])):
        x = pcg[0][example_idx][0]
        y = pcg[0][example_idx][1]
        # Ensure proper data format
        signal = np.asarray(x).flatten()
        peak_locs = np.asarray(y).flatten()
        yield signal, peak_locs


def bandpass_filter(data, low_cutoff, high_cutoff, fs, order=2):
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype="bandpass")
    y = filtfilt(b, a, data)
    return y


def peak_intervals(peak_locs, num_samples, fs, ratio_left, ratio_right):
    intervals = []
    for a,b in zip(peak_locs[:-1], peak_locs[1:]):
        window = b - a
        start_loc = a - window*ratio_left
        end_loc = a + window*ratio_right

        # Ensure indices are within signal boundaries
        if start_loc < 0:
            continue
        if end_loc > num_samples - 1:
            continue
        # Convert to time
        start_time = start_loc / fs
        end_time = end_loc / fs
        intervals.append((start_time, end_time))
    return intervals

def get_spectrogram(signal, fs, windows_per_sec):
    # Filtering params
    low_cutoff = 30  # Low cutoff frequency in Hz
    high_cutoff = 70  # High cutoff frequency in Hz
    # Spectrogram params
    nperseg = fs / windows_per_sec
    noverlap = int(nperseg / 2)  # 50% overlap
    window = 'hann'  # Hann window

    filtered_signal = bandpass_filter(signal, low_cutoff, high_cutoff, fs)
    f, t_spec, Sxx = spectrogram(
        filtered_signal,
        fs=fs,
        window="hann",
        nperseg=int(nperseg),
        noverlap=noverlap,
        scaling='density',
        mode='magnitude'
    )
    f = f[:int(100 / 5)]
    assert f[-1] == 95
    Sxx = Sxx[:20, :]  # Only up to 100 Hz

    t_spec_extended = np.insert(t_spec, 0, 0)
    f_extended = np.append(f, f[-1] + (f[1] - f[0]))

    return f_extended, t_spec_extended, np.log10(Sxx)


def compute_overlaps(xs, ys, window_size):
    overlaps = np.zeros(len(xs))
    i, j = 0, 0  # Initialize pointers for xs and ys

    while i < len(xs) and j < len(ys):
        x_start, x_end = xs[i]
        y_start, y_end = ys[j]

        # Find the overlap between xs[i] and ys[j]
        start_overlap = max(x_start, y_start)
        end_overlap = min(x_end, y_end)

        # Check if there is an actual overlap
        if start_overlap < end_overlap:
            overlaps[i] += (end_overlap - start_overlap) * window_size

        # Move the pointer that has the interval ending first
        if x_end <= y_end:
            i += 1
        else:
            j += 1

    return overlaps


def yield_samples():
    for signal, peak_locs in yield_raw_data():
        f, t, S = get_spectrogram(signal, fs, windows_per_sec)
        t_intervals = list(zip(t[:-1], t[1:]))
        peak_time_intervals = peak_intervals(peak_locs, len(signal), fs, interval_ratio_left, interval_ratio_right)
        overlaps = compute_overlaps(t_intervals, peak_time_intervals, windows_per_sec * 2)  # *2 due to overlap
        yield from zip(S.T, overlaps.reshape([-1, 1]))


def plot_some_data():
    signal, peak_locs = next(yield_raw_data())
    f, t, S = get_spectrogram(signal, fs, windows_per_sec)
    t_intervals = list(zip(t[:-1], t[1:]))
    peak_time_intervals = peak_intervals(peak_locs, len(signal), fs, interval_ratio_left, interval_ratio_right)
    overlaps = compute_overlaps(t_intervals, peak_time_intervals, windows_per_sec * 2)  # *2 due to overlap

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 10),
                                        sharex=True, gridspec_kw={'height_ratios': [5, 1, 3]})

    ax1.pcolormesh(
        t, f, S,
        shading='flat',          # Use 'flat' shading to show cell edges
        cmap='viridis',
        edgecolors='white',           # Set edge colors for grid lines
        linewidth = 0.05,  # Thin lines
    )
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_title('Spectrogram of Filtered PCG Signal at fs = 3,000 Hz (Up to 100 Hz)')
    ax1.legend(loc='upper right')

    time = np.arange(len(signal)) / fs  # Time in seconds

    # Plot the original signal on ax2
    ax3.plot(time, signal, label='Original Signal', color='blue', alpha=0.7)
    ax3.set_xlabel('Time [sec]')
    ax3.set_ylabel('Amplitude')
    ax3.set_title('Original PCG Signal with Peaks')
    ax3.set_xlim(t.min(), t.max())  # Align with spectrogram's time axis

    # Overlay peaks as vertical dashed red lines on spectrogram
    for idx, (peak_start, peak_end) in enumerate(peak_time_intervals):
        ax1.axvline(x=peak_start, color='green', linestyle='--', linewidth=1.5)
        ax1.axvline(x=peak_end, color='red', linestyle='--', linewidth=1.5)
        ax3.axvline(x=peak_start, color='green', linestyle='--', linewidth=1.5)
        ax3.axvline(x=peak_end, color='red', linestyle='--', linewidth=1.5)
    for idx, peak in enumerate(peak_locs):
        ax1.axvline(x=(peak / fs), color='blue', linestyle='--', linewidth=1.5)
        ax3.axvline(x=(peak / fs), color='blue', linestyle='--', linewidth=1.5)

    ax2.scatter((t[:-1] + t[1:]) / 2, overlaps)

    # Enhance layout and display the plot
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # High-level params
    fs = 3000  # Sampling frequency in Hz
    windows_per_sec = 5
    interval_ratio_left = 0.1
    interval_ratio_right = 0.1

    # Example 1:
    # plot_some_data()

    # Example 2:
    # data, target = next(yield_samples())
    # print(data.shape, target.shape)

    # Example 3:
    examples = list(yield_samples())
    print("Number of examples", len(examples))
    with open("dataset.pickle", "wb") as f:
        pickle.dump(examples, f)



