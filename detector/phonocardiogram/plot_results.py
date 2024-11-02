# Example 4: Make plot with the predictions
import torch

from detector.phonocardiogram.learn_net import MLP
from detector.phonocardiogram.make_dataset import plot_some_data

if __name__ == '__main__':
    with open("models/model_colab2.torch", "rb") as f:
        model = torch.load(f, map_location=torch.device("cpu"))
    plot_some_data(model)
