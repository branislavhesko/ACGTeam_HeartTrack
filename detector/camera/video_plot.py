import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import firwin, filtfilt


def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return np.array(frames)


def extract_roi(frames):
    center_x = frames.shape[2] // 2
    center_y = frames.shape[1] // 2 # -3
    roi = frames[:, center_y - 50:center_y + 50, center_x - 50:center_x + 50, 1]  # green channel
    return roi, (center_x, center_y)


def min_max_normalization(signal):
    return (signal - signal.min()) / (signal.max() - signal.min())


def process_video(video_path, model_path):
    # Load video and extract ROI
    frames = load_video(video_path)
    roi, (center_x, center_y) = extract_roi(frames)

    # Calculate signal
    signal = np.mean(roi, axis=(1, 2))
    # signal = np.mean(frames[:, :, :, 1], axis=(1, 2))
    signal = min_max_normalization(signal)

    return signal, frames, (center_x, center_y)


def plot_results(signal, frames, center_coords):
    # Plot original vs denoised signal
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(signal, label='Original Signal')
    plt.subplot(2, 1, 2)
    filter = firwin(10, 100, fs=30)
    filtered_signal = filtfilt(filter, 1, signal)
    plt.plot(filtered_signal, label='Filtered Signal')
    plt.legend()
    plt.show()

    # Plot ROI on first frame
    frame = frames[0].copy()
    center_x, center_y = center_coords
    cv2.rectangle(frame,
                  (center_x - 30, center_y - 30),
                  (center_x + 30, center_y + 30),
                  (0, 255, 0), 2)
    plt.figure()
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Frame with ROI')
    plt.show()


if __name__ == "__main__":
    VIDEO_PATH = '2024-11-02_15-10-32_REC1464131401588062562.mp4'
    MODEL_PATH = 'C:/Users/vojta/PycharmProjects/ACGTeam_HeartTrack/detector/denoising/checkpoints/model_25.pt'

    signal, frames, center_coords = process_video(VIDEO_PATH, MODEL_PATH)
    plot_results(signal, frames, center_coords)
