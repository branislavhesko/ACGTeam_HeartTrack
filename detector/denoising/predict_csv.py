import os

import torch
import torch.nn as nn

from model import UNet

import matplotlib.pyplot as plt


csv_path = "C:/Users/vojta/Downloads/csvs"
MODEL_PATH = 'C:/Users/vojta/PycharmProjects/ACGTeam_HeartTrack/detector/denoising/checkpoints/model_36.pt'


checkpoint = torch.load(MODEL_PATH, map_location='cpu')
model = UNet()
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

for file in os.listdir(csv_path):
    if file.endswith(".csv"):
        print(file)
        with open(os.path.join(csv_path, file), 'r') as f:
            lines = f.readlines()
            signal = [float(line.strip()) for line in lines]
            signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            quality_output = model(signal)
            print(f"Quality: {torch.argmax(quality_output, dim=1)}")
            print(f"Quality: {quality_output}")

            plt.figure()
            plt.plot(signal.squeeze().numpy())
            plt.title(f"Quality: {torch.argmax(quality_output, dim=1).item()}")
            plt.show()
