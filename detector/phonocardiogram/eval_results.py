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
from itertools import islice

import numpy as np
import torch

from detector.phonocardiogram.learn_net import MLP, predictions_to_peaks, eval_sequence
from detector.phonocardiogram.make_dataset import plot_some_data, yield_raw_data, yield_full_sequences, describe


def evaluate_bpm(model):
    bpm_errors = []
    dist_errors = []
    for idx, data in enumerate(yield_full_sequences()):
        norm_data = data[0]
        overlaps = data[1]
        t = data[2]
        peak_times = data[3]

        threshold = 0.01
        with torch.no_grad():
            y_pred = model(torch.from_numpy(norm_data).to(torch.float32)).numpy()

        binary_preds = (y_pred > threshold).astype(int).flatten()
        binary_targets = (overlaps > threshold).astype(int).flatten()
        mask = (binary_preds == binary_targets)
        t_mid = (t[:-1] + t[1:]) / 2

        y_peaks = list(predictions_to_peaks(t, y_pred, threshold))
        pred_distance = eval_sequence(peak_times, y_peaks)
        actual_bpm = len(peak_times) / max(t) * 60
        predicted_bpm = len(y_peaks) / max(t) * 60
        bpm_errors.append(abs(actual_bpm - predicted_bpm))
        dist_errors.append(pred_distance)
        print(idx, "BPM", actual_bpm, "predicted", predicted_bpm, " abs distance", pred_distance)

    print("Very bad sequences")
    for idx, err in enumerate(bpm_errors):
        if err > 40:
            print(idx)

    print("BPM stats")
    describe(bpm_errors)
    print("Distance stats")
    describe(dist_errors)


def eval_mp4_file(mp4_file, model):
    norm_data, overlaps, t, f, peak_times, signal, peak_time_intervals = next(yield_full_sequences(file_name=mp4_file))
    with torch.no_grad():
        threshold = 0.01
        y_pred = model(torch.from_numpy(norm_data).to(torch.float32)).numpy()
        predicted_peaks = list(predictions_to_peaks(t, y_pred, threshold))

    # All numpy arrays, norm_data is 2d array, otherwise 1d arrays
    return t, f, norm_data, predicted_peaks

def plot_evaluation(t, f, norm_data, predicted_peaks):
    fig, ax1 = plt.subplots(1, 1, figsize=(14, 5))
    ax1.pcolormesh(
        t, f, norm_data.T,
        shading='flat',  # Use 'flat' shading to show cell edges
        cmap='viridis',
        edgecolors='white',  # Set edge colors for grid lines
        linewidth=0.05,  # Thin lines
    )
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_ylabel('Time [sec]')
    ax1.set_title('Spectrogram of Filtered PCG Signal (Up to 100 Hz)')
    ax1.legend(loc='upper right')
    for idx, peak in enumerate(predicted_peaks):
        ax1.axvline(x=peak, color='red', linestyle='--', linewidth=1.5)
    plt.show()

if __name__ == '__main__':
    with open("models/model_colab3.torch", "rb") as f:
        model = torch.load(f, map_location=torch.device("cpu"))

    # plot_some_data(model, file_name="data/2024-11-02_19-33-19_REC5893019824290697924.mp4")
    # evaluate_bpm(model)

    t, f, norm_data, predicted_peaks = eval_mp4_file(model=model, mp4_file="data/2024-11-02_19-33-19_REC5893019824290697924.mp4")
    plot_evaluation(t, f, norm_data, predicted_peaks)
    print("BPM", (len(predicted_peaks) / max(t)) * 60)
