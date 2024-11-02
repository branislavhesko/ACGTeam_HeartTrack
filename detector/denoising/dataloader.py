import dataclasses
import pathlib

import pandas as pd
import torch
import wfdb


class GaussianNoise(torch.nn.Module):
    def __init__(self, mean: float, std: float, probability: float):
        super().__init__()
        self.probability = probability
        self.mean = mean
        self.std = std

    def forward(self, x: torch.Tensor):
        if torch.rand(1) < self.probability:
            return x + torch.randn_like(x) * self.std + self.mean
        else:
            return x
        

class MotionBlur(torch.nn.Module):
    def __init__(self, frequency_min: float, frequency_max: float, probability: float):
        super().__init__()
        self.frequency_min = frequency_min
        self.frequency_max = frequency_max
        self.probability = probability
        
    def forward(self, x: torch.Tensor):
        pass


class PoissonNoise(torch.nn.Module):
    def __init__(self, lam: float, probability: float):
        super().__init__()
        self.lam = lam
        self.probability = probability

    def forward(self, x: torch.Tensor):
        if torch.rand(1) < self.probability:
            return x + torch.poisson(torch.randn_like(x) * self.lam)
        else:
            return x


class Compose(torch.nn.Module):
    def __init__(self, transforms: list[torch.nn.Module]):
        super().__init__()
        self.transforms = torch.nn.ModuleList(transforms)

    def forward(self, x: torch.Tensor):
        for transform in self.transforms:
            x = transform(x)
        return x


def load_record(record_name):
    record = wfdb.rdrecord(record_name)
    return record


def min_max_normalization(signal):
    signal = signal - signal.min()
    signal = signal / signal.max()
    return signal


class WFDBDataset(torch.utils.data.Dataset):
    def __init__(self, record_names, quality_df, transform: torch.nn.Module):
        self.record_names = record_names
        self.quality_df = quality_df
        self.quality_annotation = self._make_quality_annotation()
        self.transform = transform

    def _make_quality_annotation(self):
        quality_annotation = {}
        for _, row in self.quality_df.iterrows():
            quality_annotation[row["ID"]] = row["Quality"]
        return quality_annotation

    def __len__(self):
        return len(self.record_names)

    def __getitem__(self, idx):
        record_name = self.record_names[idx]
        record_folder_name = pathlib.Path(record_name).parent.name
        record = load_record(record_name)
        quality = self.quality_annotation[int(record_folder_name)]
        data = torch.from_numpy(record.p_signal).float()

        data = data[torch.randint(0, data.shape[0], (1,)), :]

        return min_max_normalization(data).float(), self.transform(min_max_normalization(data)).float(), torch.tensor(quality).long()


def filter_bad_data(ppg_file, assert_length_min: int = 300):
    record = load_record(ppg_file.with_suffix(""))
    if record.p_signal.shape[1] < assert_length_min:
        return False
    return True


def get_dataloader(
        folder_path: str | pathlib.Path,
        batch_size: int,
        num_workers: int,
        pin_memory: bool
):
    ppg_files = list(pathlib.Path(folder_path).rglob("*PPG.dat"))
    ppg_files = [file for file in ppg_files if filter_bad_data(file)]
    ppg_files_without_ext = [file.with_suffix("") for file in ppg_files]
    quality_file = pathlib.Path(folder_path) / "quality-hr-ann.csv"
    quality_df = pd.read_csv(quality_file)
    transform = Compose([
        GaussianNoise(mean=0, std=0.005, probability=0.5),
    ])
    dataset = WFDBDataset(ppg_files_without_ext, quality_df, transform)
    return torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
        pin_memory=pin_memory,
        drop_last=True,
        persistent_workers=True
    )


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    dataloader = get_dataloader(
        # "/Users/brani/Downloads/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0",
        "C:/Users/vojta/Downloads/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0",
        batch_size=1,
        num_workers=1,
        pin_memory=True
    )
    for record, quality in dataloader:
        print(record)
        for i in range(record.shape[-1]):
            plt.plot(record[0, :, i].numpy())
        plt.title(f"Quality: {quality[0]}")
        plt.show()