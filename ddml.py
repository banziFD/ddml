import numpy as np  
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

class DDMLRes(nn.Module):
    def __init__(self):
        super(DDML, self).__init__()
        self.feature = models.resnet18(pretrained = True)
        self.feature = nn.Sequential(*list(self.feature.children())[:-1])
        if (pretrained != None):
            self.feature.load_state_dict(pretrained)
            print('Complete load pretrained information')
        self.length = 512
    
    def  forward(self, x):
        y = self.feature(x)
        y = y.reshape(-1, 512)
        return y

class DDML(nn.Module):
    def __init__(self, pretrained = None):
        super(DDML, self).__init__()
        self.feature = models.resnet18(pretrained = True)
        self.feature = nn.Sequential(*list(self.feature.children())[:-1])
        if (pretrained != None):
            self.feature.load_state_dict(pretrained)
            print('Complete load pretrained information')
      
    def  forward(self, x):
        y = self.feature(x)
        y = y.reshape(-1, 512)
        return y

class DDMLLoss(nn.Module):
    def __init__(self, tau = 2, beta = 1):
        super(DDMLLoss, self).__init__()
        self.softMargin = nn.SoftMarginLoss()
        self.tau = tau
        self.beta = beta
        self.relu = nn.ReLU()

    def forward(self, y1, y2, sim):
        sim = torch.tensor(sim).float()
        sim = sim * 2 - 1
        y = y1 - y2
        assert y.shape[0] == sim.shape[0]
        distance = torch.norm(y, 2, 1)
        loss = 1 - sim * (self.tau - distance)
        loss = self.relu(loss)
        loss = 1 / self.beta * (torch.log(1 + torch.exp(self.beta * loss)))
        loss = loss.sum() / y.shape[0]
        return loss


if __name__ == "__main__": 
    x = torch.rand(10, 3, 32, 32)
    m = ConvFeature()
    y = m(x)
    print(y.shape)