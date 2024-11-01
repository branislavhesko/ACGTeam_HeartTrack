import dataclasses
import pathlib

import pandas as pd
import torch
import wfdb


def load_record(record_name):
    record = wfdb.rdrecord(record_name)
    return record


def min_max_normalization(signal):
    signal = signal - signal.min()
    signal = signal / signal.max()
    return signal


class WFDBDataset(torch.utils.data.Dataset):
    def __init__(self, record_names, quality_df):
        self.record_names = record_names
        self.quality_df = quality_df
        self.quality_annotation = self._make_quality_annotation()
        
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
        return min_max_normalization(data).float(), torch.tensor(quality).long()


def get_dataloader(
        folder_path: str | pathlib.Path,
        batch_size: int,
        num_workers: int,
        pin_memory: bool 
    ):
    ppg_files = list(pathlib.Path(folder_path).rglob("*PPG.dat"))
    ppg_files_without_ext = [file.with_suffix("") for file in ppg_files]
    quality_file = pathlib.Path(folder_path) / "quality-hr-ann.csv"
    quality_df = pd.read_csv(quality_file)
    dataset = WFDBDataset(ppg_files_without_ext, quality_df)
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
        "/Users/brani/Downloads/brno-university-of-technology-smartphone-ppg-database-but-ppg-2.0.0",
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