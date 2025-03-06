import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random

from cbam import CBAM
from inception import InceptionModule

class MixStyle(nn.Module):

    def __init__(self, p=0.0, alpha=0.1, eps=1e-6, mix='random'):
        super().__init__()
        self.p = p
        self.beta = torch.distributions.Beta(alpha, alpha)
        self.eps = eps
        self.alpha = alpha
        self.mix = mix
        self._activated = True

    def __repr__(self):
        return f'MixStyle(p={self.p}, alpha={self.alpha}, eps={self.eps}, mix={self.mix})'

    def set_activation_status(self, status=True):
        self._activated = status

    def update_mix_method(self, mix='random'):
        self.mix = mix

    def forward(self, x):
        if not self.training or not self._activated:
            return x

        if random.random() > self.p:
            return x

        B = x.size(0)

        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True)
        sig = (var + self.eps).sqrt()
        mu, sig = mu.detach(), sig.detach()
        x_normed = (x-mu) / sig

        lmda = self.beta.sample((B, 1, 1, 1))
        lmda = lmda.to(x.device)

        if self.mix == 'random':
            # random shuffle
            perm = torch.randperm(B)#随机生成一个0~B-1的数字

        elif self.mix == 'crossdomain':
            # split into two halves and swap the order
            perm = torch.arange(B - 1, -1, -1) # inverse index
            perm_b, perm_a = perm.chunk(2)
            perm_b = perm_b[torch.randperm(math.ceil(B / 2))]
            perm_a = perm_a[torch.randperm(B // 2)]
            perm = torch.cat([perm_b, perm_a], 0)

        else:
            raise NotImplementedError

        mu2, sig2 = mu[perm], sig[perm]
        mu_mix = mu*lmda + mu2 * (1-lmda)
        sig_mix = sig*lmda + sig2 * (1-lmda)

        return x_normed*sig_mix + mu_mix


class WaveletBasedResidualAttentionNet(nn.Module):
    def __init__(self, input_channels=4, depth=1, ratio=4, width=64, alpha=0.01, num_classes=7):
        super().__init__()
        self.depth = depth

        self.feature_extraction = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=64, kernel_size=7,  padding=3),
            nn.LeakyReLU(negative_slope=alpha)
        )

        self.inception_module = InceptionModule(in_channels=width, width=width, ratio=ratio, alpha=alpha)
        self.mixstyle = MixStyle(p=0.7, alpha=0.1)

        self.final_layers = nn.Sequential(
            nn.Conv2d(in_channels=width * depth, out_channels=width, kernel_size=3, padding=1),
            nn.LeakyReLU(negative_slope=alpha),
            nn.Conv2d(in_channels=width, out_channels=4, kernel_size=3, padding=1),
        )
        self.relu = nn.ReLU()
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.LazyLinear(1000))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(1000))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout())
        self.class_classifier.add_module('c_fc2', nn.Linear(1000, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 7))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax(dim=1))  # 将输入n维向量缩放到（0，1）之间且和为1
    def forward(self, x):
        out = self.feature_extraction(x)
        outs = []
        for _ in range(self.depth):
            residual = out
            out = self.inception_module(out)
            out = self.mixstyle(out)
            out += residual
            outs.append(out)
        out = self.final_layers(torch.cat(outs, dim=1))
        out = self.mixstyle(out)
        out = out.view(out.size(0), -1)#将张量展平，第一个维度是批次，后面的三维展开为一维
        #18360
        out = self.class_classifier(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                # Xavier's initialization for convolutional layers
                init.xavier_uniform_(tensor=m.weight, gain=0.1)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

def my_resnet():
    model = WaveletBasedResidualAttentionNet(4, 2, 4, 64, 0.01, 7)
    return model