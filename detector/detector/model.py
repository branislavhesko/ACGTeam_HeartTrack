import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv1d(in_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU(),
            nn.Conv1d(out_c, out_c, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_c),
            nn.ReLU()
        )

    def forward(self, inputs):
        return self.layers(inputs)


class EncoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = ConvBlock(in_c, out_c)
        self.pool = nn.MaxPool1d(2)

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class DecoderBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose1d(in_c, out_c, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_c + out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        if x.size(2) != skip.size(2):
            diff = skip.size(2) - x.size(2)
            x = nn.functional.pad(x, (diff // 2, diff - diff // 2))
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        self.encoder = nn.ModuleList([
            EncoderBlock(in_channels, 16),
            EncoderBlock(16, 32),
            EncoderBlock(32, 64),
            EncoderBlock(64, 128)
        ])
        self.bottleneck = ConvBlock(128, 256)
        self.decoder = nn.ModuleList([
            DecoderBlock(256, 128),
            DecoderBlock(128, 64),
            DecoderBlock(64, 32),
            DecoderBlock(32, 16)
        ])
        self.outputs = nn.Conv1d(16, out_channels, kernel_size=1)

    def forward(self, inputs):
        skips = []
        x = inputs
        for enc in self.encoder:
            x, p = enc(x)
            skips.append(x)
            x = p

        x = self.bottleneck(x)


        for dec, skip in zip(self.decoder, reversed(skips)):
            x = dec(x, skip)

        return self.outputs(x)
    

if __name__ == "__main__":
    model = UNet()
    from tqdm import tqdm
    for i in tqdm(range(10000)):
        x = torch.randn(1, 1, 300)
        y = model(x)
