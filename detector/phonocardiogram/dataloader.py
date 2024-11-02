import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.signal import butter, filtfilt
from scipy.io import loadmat
import matplotlib.pyplot as plt

fs = 3000 
low_cutoff = 30 
high_cutoff = 80  

def bandpass_filter(data, low_cutoff, high_cutoff, fs, order=2):
    nyquist = 0.5 * fs
    low = low_cutoff / nyquist
    high = high_cutoff / nyquist
    b, a = butter(order, [low, high], btype="band")
    y = filtfilt(b, a, data)
    return y

class PCGDataset(Dataset):
    def __init__(self, pcg_data, transform=None):
        self.pcg_data = pcg_data
        self.transform = transform

    def __len__(self):
        return len(self.pcg_data)

    def __getitem__(self, idx):
        signal = self.pcg_data[idx][0]
        peak_locs = self.pcg_data[idx][1]
        
        filtered_signal = bandpass_filter(signal.flatten(), low_cutoff, high_cutoff, fs)
        
        if self.transform:
            filtered_signal = torch.tensor(filtered_signal, dtype=torch.float32)
            peak_locs = torch.tensor(peak_locs, dtype=torch.float32)

        return filtered_signal, peak_locs

class ToTensor:
    def __call__(self, sample):
        signal, peak_locs = sample['signal'], sample['peak_locs']
        return {'signal': torch.tensor(signal, dtype=torch.float32),
                'peak_locs': torch.tensor(peak_locs, dtype=torch.float32)}

# Load the PCG dataset
pcg = loadmat('C:/Users/vojta/Downloads/PCG_dataset.mat')['PCG_dataset']

# Assuming pcg is a numpy array
pcg_data = np.array([pcg[0, 8]])  
transformed_dataset = PCGDataset(pcg_data=pcg_data, transform=ToTensor())

dataloader = DataLoader(transformed_dataset, batch_size=4, shuffle=True, num_workers=0)

for i_batch, sample_batched in enumerate(dataloader):
    print(i_batch, sample_batched['signal'].size(), sample_batched['peak_locs'].size())

# Plotting the filtered signal for the first batch
for i_batch, sample_batched in enumerate(dataloader):
    plt.plot(sample_batched['signal'][0].numpy()[:20000])
    break
plt.show()