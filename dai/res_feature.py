import torch.nn as nn
import torchvision.models as models


class ResFeature(nn.Module):
    def __init__(self):
        super(ResFeature, self).__init__()
        self.feature = models.resnet152(pretrained = True)
        self.feature = nn.Sequential(*list(self.feature.children())[:-1])

    def forward(self, x):
        y = self.feature(x)
        y = y.reshape(-1, 2048)
        return y