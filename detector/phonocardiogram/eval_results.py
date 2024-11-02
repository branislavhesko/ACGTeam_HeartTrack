# Example 4: Make plot with the predictions
from itertools import islice

import numpy as np
import torch

from detector.phonocardiogram.learn_net import MLP, predictions_to_peaks, eval_sequence
from detector.phonocardiogram.make_dataset import plot_some_data, yield_raw_data, yield_full_sequences, describe


def evaluate_bpm(model):
    bpm_errors = []
    dist_errors = []
    for idx, (norm_data, overlaps, t, peak_times) in enumerate(yield_full_sequences()):
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

if __name__ == '__main__':
    with open("models/model_colab3.torch", "rb") as f:
        model = torch.load(f, map_location=torch.device("cpu"))

    plot_some_data(model, example_idx=597)
    # evaluate_bpm(model)
