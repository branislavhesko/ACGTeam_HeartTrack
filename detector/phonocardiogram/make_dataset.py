import os.path
import pickle
import json
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from scipy.signal import butter, filtfilt, spectrogram, lfilter, resample

from detector.phonocardiogram.learn_net import predictions_to_peaks, eval_sequence
import statistics

from pydub import AudioSegment
from scipy.fft import rfft, rfftfreq, irfft
import numpy as np


def describe(data):
    if not data:
        print("The list is empty.")
        return

    print("Statistics of the list:")
    print(f"Count: {len(data)}")
    print(f"Min: {min(data)}")
    print(f"Max: {max(data)}")
    print(f"Mean: {statistics.mean(data):.2f}")
    print(f"Median: {statistics.median(data):.2f}")
    print(f"Standard Deviation: {statistics.stdev(data):.2f}" if len(
        data) > 1 else "Standard Deviation: Not applicable (need at least 2 values)")


_PCG = None

BAD_SEQUENCES = [
84,
        163,
        206,
        233,
        278,
        316,
        346,
        538,
        544,
        597,
]

def yield_video(file_name: str):
    signal = load_and_filter_audio(file_name)
    peak_times = []
    return signal, peak_times


def yield_ref_data(example_idx):
    if example_idx in BAD_SEQUENCES:
        return
    pcg = _PCG
    assert pcg is not None
    x = pcg[0][example_idx][0]
    y = pcg[0][example_idx][1]
    # Ensure proper data format
    signal = np.asarray(x).flatten()
    peak_locs = np.asarray(y).flatten()
    peak_times = peak_locs / fs
    return signal, peak_times


def yield_raw_data(example_idx = None, file_name = None):
    # global _PCG
    # if _PCG is not None:
    #     pcg = _PCG
    # else:
    #     mat = loadmat('PCG_dataset.mat')
    #     pcg = mat['PCG_dataset']
    #     _PCG = pcg

    if file_name:
        yield yield_video(file_name)
    elif example_idx is not None:
        data = yield_ref_data(example_idx)
        if data:
            yield data
    else:
        for example_idx in range(len(pcg[0])):
            data = yield_ref_data(example_idx)
            if data:
                yield data



def load_and_filter_audio(mp4_file):
    # Load audio from the MP4 file
    print(mp4_file)
    audio = AudioSegment.from_file(mp4_file, format="mp4")

    # Convert to mono if necessary
    if audio.channels > 1:
        audio = audio.set_channels(1)

    # Extract raw audio data as a NumPy array
    samples = np.array(audio.get_array_of_samples())

    # Convert samples to float for processing
    samples = samples.astype(np.float32)

    # Original sampling rate
    sample_rate = audio.frame_rate

    # Define the cutoff frequency and normalize it
    cutoff_freq = 1500  # 1.5 kHz
    nyquist_rate = 0.5 * sample_rate
    normalized_cutoff = cutoff_freq / nyquist_rate

    # Create a low-pass Butterworth filter
    b, a = butter(N=5, Wn=normalized_cutoff, btype='low', analog=False)

    # Apply the filter to the audio samples
    filtered_samples = lfilter(b, a, samples)

    # Resample the audio to 3000 Hz
    target_sample_rate = 3000  # 3000 samples per second
    num_samples = int(len(filtered_samples) * target_sample_rate / sample_rate)
    resampled_samples = resample(filtered_samples, num_samples)

    return np.asarray(resampled_samples).flatten()


def load_peak_times(json_file):
    if not os.path.exists(json_file):
        print("No reference peak times found! File is missing. ", json_file)
        return []
    def extract_times(data):
        # Extract times from the points list
        times = [point["time"] for point in data["points"]]
        return times

    with open(json_file, "r") as f:
        return extract_times(json.load(f))



def bandpass_filter(data, low_cutoff, high_cutoff, fs, order=2):
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype="bandpass")
    y = filtfilt(b, a, data)
    return y


def peak_intervals(peak_times, total_time, fs, ratio_left, ratio_right):
    intervals = []
    for a,b in zip(peak_times[:-1], peak_times[1:]):
        window = b - a
        start_loc = a - window*ratio_left
        end_loc = a + window*ratio_right

        # Ensure indices are within signal boundaries
        if start_loc < 0:
            continue
        if end_loc > total_time - 1:
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


def yield_full_sequences(example_idx = None, file_name = None):
    for signal, peak_times in yield_raw_data(example_idx, file_name):
        f, t, S = get_spectrogram(signal, fs, windows_per_sec)
        t_intervals = list(zip(t[:-1], t[1:]))
        peak_time_intervals = peak_intervals(peak_times, max(t), fs, interval_ratio_left, interval_ratio_right)
        overlaps = compute_overlaps(t_intervals, peak_time_intervals, windows_per_sec * 2)  # *2 due to overlap
        S = S.T
        norm_data = (S - np.mean(S, axis=0))
        norm_data /= np.std(norm_data, axis=0)
        yield norm_data, overlaps, t, f, peak_times, signal, peak_time_intervals


def yield_samples():
    for norm_data, overlaps, t, f, peak_times, signal, peak_time_intervals in yield_full_sequences():
        for x, y in zip(norm_data, overlaps):
            yield x.ravel().reshape(-1, 1), y.ravel().reshape(-1, 1)


def plot_some_data(model = None, example_idx = None, file_name = None, save = False):
    norm_data, overlaps, t, f, peak_times, signal, peak_time_intervals = next(yield_full_sequences(example_idx, file_name))
    fig, (ax1, ax2, ax4, ax3) = plt.subplots(4, 1, figsize=(14, 10),
                                        sharex=True, gridspec_kw={'height_ratios': [5, 1, 1, 3]})

    ax1.pcolormesh(
        t, f, norm_data.T,
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
    for idx, peak in enumerate(peak_times):
        ax1.axvline(x=peak, color='blue', linestyle='--', linewidth=1.5)
        ax3.axvline(x=peak, color='blue', linestyle='--', linewidth=1.5)

    ax2.scatter((t[:-1] + t[1:]) / 2, overlaps)

    if model:
        threshold = 0.01
        with torch.no_grad():
            y_pred = model(torch.from_numpy(norm_data).to(torch.float32)).numpy()

        binary_preds = (y_pred > threshold).astype(int).flatten()
        t_mid = (t[:-1] + t[1:]) / 2
        ax4.plot(t_mid, y_pred)
        ax4.plot([t_mid[0], t_mid[-1]], [threshold, threshold], "k--")

        binary_targets = (overlaps > threshold).astype(int).flatten()
        mask = (binary_preds == binary_targets)
        ax4.scatter(t_mid[mask],  y_pred[mask], color="g")
        ax4.scatter(t_mid[~mask], y_pred[~mask], color="r")

    # Enhance layout and display the plot
    plt.tight_layout()
    if save:
        print(f"Saving sequences/{example_idx}.jpg")
        plt.savefig(f"sequences/{example_idx}.jpg")
    else:
        plt.show()



# High-level params
fs = 3000  # Sampling frequency in Hz
windows_per_sec = 5
interval_ratio_left = 0.1
interval_ratio_right = 0.1


if __name__ == '__main__':
    # Example 1: Make a basic plot
    # plot_some_data()

    # Example 2: Get a single example from dataset
    # data, target = next(yield_samples())
    # print(data, target)
    # print(data.shape, target.shape)

    # Example 3: Process whole dataset
    examples = list(yield_samples())
    print("Number of examples", len(examples))
    with open("data.pkl", "wb") as f:
        pickle.dump(examples, f)
