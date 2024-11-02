import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch


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
    return (signal - signal.min()) / signal.max()


def process_video(video_path, model_path):
    # Load video and extract ROI
    frames = load_video(video_path)
    roi, (center_x, center_y) = extract_roi(frames)

    # Calculate signal
    signal = np.mean(roi, axis=(1, 2))
    # signal = np.mean(frames[:, :, :, 1], axis=(1, 2))
    signal = min_max_normalization(signal)

    # Denoise signal
    model = torch.jit.load(model_path, map_location='cpu')
    model.eval()
    tensor = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    denoised_output, quality_output = model(tensor[:, :, :100])

    print(f"Quality: {quality_output.item()}")

    return signal, denoised_output, frames, (center_x, center_y)


def plot_results(signal, denoised_output, frames, center_coords):
    # Plot original vs denoised signal
    plt.figure()
    plt.plot(signal, label='Original Signal')
    # plt.plot(denoised_output.detach().numpy().flatten(), label='Denoised Signal')
    plt.legend()
    plt.show()

    # Plot ROI on first frame
    frame = frames[0].copy()
    center_x, center_y = center_coords
    cv2.rectangle(frame,
                  (center_x - 50, center_y - 50),
                  (center_x + 50, center_y + 50),
                  (0, 255, 0), 2)
    plt.figure()
    plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.title('Frame with ROI')
    plt.show()


if __name__ == "__main__":
    VIDEO_PATH = 'C:/Users/vojta/Downloads/PPG_android/2024-11-02_15-10-10_REC9145263651989545060.mp4'
    MODEL_PATH = 'C:/Users/vojta/PycharmProjects/ACGTeam_HeartTrack/detector/denoising/checkpoints/model_25.pt'

    signal, denoised_output, frames, center_coords = process_video(VIDEO_PATH, MODEL_PATH)
    plot_results(signal, denoised_output, frames, center_coords)
