import os

import torch
import torch.nn as nn

from model import UNet
from dataloader import min_max_normalization

import matplotlib.pyplot as plt


def load_model(model_path):
    checkpoint = torch.load(model_path, map_location='cpu')
    model = UNet()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict(model, signal):
    model.eval()
    signal = min_max_normalization(torch.tensor(signal)).float()
    signal = torch.tensor(signal, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    quality_output = model(signal)
    return quality_output


if __name__ == "__main__":
    MODEL_PATH = 'C:/Users/vojta/PycharmProjects/ACGTeam_HeartTrack/detector/denoising/checkpoints/model_36.pt'
    SIGNAL = ...
    model = load_model(MODEL_PATH)
    prediction = predict(model, SIGNAL)



