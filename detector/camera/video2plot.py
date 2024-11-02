import torch


def plot_data(signal: torch.Tensor) -> torch.Tensor:
    signal = torch.mean(signal, dim=(1, 2, 3))

    return signal


if __name__ == "__main__":
    data = torch.randn(100, 224, 224, 1)
    print(plot_data(data).shape)
