import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

from detector.detector.dataset import get_dataloader
from detector.detector.model import UNet


def min_max_normalize(x):
    return (x - x.min()) / (x.max() - x.min())

class Trainer:
    def __init__(self, path: str) -> None:
        self.dataloader = get_dataloader(path, batch_size=2)
        self.model = UNet(4, 1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-3)
        self.loss_quality = torch.nn.CrossEntropyLoss()
        self.loss_click = torch.nn.CrossEntropyLoss()
        self.writer = SummaryWriter()
        self.epoch = 0
        self.step = 0
        
    def visualize(self, ppg, click_labels, quality_labels, output):
        fig = plt.figure(dpi=200)
        plt.subplot(3, 2, 1)
        plt.plot(ppg[0].squeeze(0))
        plt.title("PPG")
        plt.subplot(3, 2, 2)
        plt.plot(click_labels[0].squeeze(0))
        plt.plot(quality_labels[0].squeeze(0))
        plt.legend(["Click", "Quality"])
        plt.title("Labels")
        plt.subplot(3, 2, 3)
        plt.plot(output[0, 0, :])
        plt.plot(output[0, 1, :])
        plt.legend(["Bad", "Good"])
        plt.title("Quality")
        plt.subplot(3, 2, 4)
        plt.plot(output[0, 2, :])
        plt.plot(output[0, 3, :])
        plt.legend(["None", "Detection"])
        plt.title("Click")
        plt.subplot(3, 2, 5)
        plt.plot(output[0, :2, :].argmax(dim=0))
        plt.title("Quality")
        plt.subplot(3, 2, 6)
        plt.plot(output[0, 2:, :].argmax(dim=0))
        plt.title("Click")
        return fig

    def train(self):
        self.model.train()
        for epoch in range(50):
            for idx, (ppg, click_labels, quality_labels) in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                output = self.model(ppg)
                quality_loss = self.loss_quality(output[:, :2, :], quality_labels)
                click_loss = 0
                for batch_idx in range(len(click_labels)):
                    if quality_labels[batch_idx].sum() > 0:
                        click_loss += self.loss_click(output[:, 2:, :][batch_idx][:,quality_labels[batch_idx] == 1].unsqueeze(0), click_labels[batch_idx][quality_labels[batch_idx] == 1].unsqueeze(0))
                loss = quality_loss + click_loss
                loss.backward()
                self.optimizer.step()
                self.writer.add_scalar("Loss/quality", quality_loss, self.step)
                self.writer.add_scalar("Loss/click", click_loss, self.step)
                self.writer.add_figure(f"Output_{idx}", self.visualize(ppg, click_labels, quality_labels, output.detach()), epoch)
                print(f"Epoch {epoch}, Batch {idx}, Loss: {loss.item()}")
                self.step += 1
        torch.jit.script(self.model).save("detector.pt")
        

if __name__ == "__main__":
    path = "/Users/brani/code/ACGTeam_HeartTrack_data/"
    trainer = Trainer(path)
    trainer.train()
