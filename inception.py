import torch
import torch.nn as nn


class InceptionModule(nn.Module):
    def __init__(self, in_channels, width, ratio=4, alpha=0.01):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, width // ratio, kernel_size=1, padding=0)
        self.conv2 = nn.Conv2d(in_channels, width // ratio, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, width // ratio, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(in_channels, width // ratio, kernel_size=7, padding=3)
        self.leaky_relu = nn.LeakyReLU(negative_slope=alpha)
        self.final_conv = nn.Conv2d(4 * (width // ratio), width, kernel_size=1, padding=0)

    def forward(self, x):
        conv1 = self.leaky_relu(self.conv1(x))
        conv2 = self.leaky_relu(self.conv2(x))
        conv3 = self.leaky_relu(self.conv3(x))
        conv4 = self.leaky_relu(self.conv4(x))
        concatenated = torch.cat([conv1, conv2, conv3, conv4], dim=1)

        out = self.final_conv(concatenated)

        return self.leaky_relu(out)
