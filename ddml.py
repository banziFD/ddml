import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR as MultiStepLR
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from visdom import Visdom
import util.data.cifar as cifar

class ResFeature(nn.Module):
    def __init__(self):
        super(ResFeature, self).__init__()
        self.feature = models.resnet18(pretrained = True)
        self.feature = nn.Sequential(*list(self.feature.children())[:-1])
        self.length = 512
    
    def forward(self, x):
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

class DDML:
    def __init__(self, pa, featureNet, workPath, loader):
        self.featureNet = featureNet
        self.workPath = workPath
        self.pa = pa
        self.loader = loader

    def __makeHook__(flag, data):
        if flag == 'f':
            def hook(m, input, output):
                data.append(input)
            return hook
        if flag == 'b':
            def hook(m, input, output):
                data.append(output)
            return hook
    
    def __train__(self, pa, workPath, ddml, trainLoader1, trainLoader2, valLoader1, valLoader2, lossFun, optim):
        vis = Visdom()
        curveX = torch.zeros(pa['epoch'] * 2, 2)
        curveY = torch.zeros(pa['epoch'] * 2)
        curve = 0
        if len(pa['milestones']) != 0:
            scheduler = MultiStepLR(optim, pa['milestones'], gamma = 0.1)
        else:
            scheduler = None
           
        for e in range(pa['epoch']):
            if scheduler:
                scheduler.step()
            start = time.time()
            trainError = 0
            valError = 0

            slideCurveX = torch.zeros(len(trainLoader1), 2)
            freqError = 0
            freqStep = 0
            slideCurveY = torch.zeros(len(trainLoader1))

            ddml.train()
            it = trainLoader2.__iter__()
            for step, (x1, label1) in enumerate(trainLoader1):
                (x2, label2) = it.__next__()
                sim = (label1 == label2)
                # if(step == 0):
                #     inter_feature = []
                #     handle1 = ddml.feature.linear1.register_forward_hook(make_hook('f', inter_feature))
                #     handle2 = ddml.feature.linear2.register_forward_hook(make_hook('f', inter_feature))
                #     handle3 = ddml.feature.linear3.register_forward_hook(make_hook('f', inter_feature))
                x1.requires_grad = False;
                x2.requires_grad = False;
                if(pa['gpu']):
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    sim = sim.cuda()
                    
                y1 = ddml(x1)
                y2 = ddml(x2)
                y2 = y2.detach()
                optim.zero_grad()
                loss = lossFun(y1, y2, sim)

                # l2_reg = torch.tensor(0.0).cuda()
                # for p in ddml.parameters():
                #     l2_reg += torch.norm(p)
                # loss += pa['lambda']*l2_reg

                loss.backward()
                optim.step()

                trainError += loss.data.item()
                freqError += loss.data.item()

                # if step == 0:
                #     inter_feature = [item[0][0:10] for item in inter_feature]
                #     temp = inter_feature[5]
                #     inter_feature = inter_feature[0:3]
                #     inter_feature[2] = inter_feature[2] - temp
                #     inter_feature[0] = inter_feature[0].reshape(-1, 3, 32, 32)
                #     inter_feature[1] = inter_feature[1].reshape(-1, 1, 16, 16)
                #     inter_feature[2] = inter_feature[2].reshape(-1, 1, 16, 16)
                #     vis.images(inter_feature[0], nrow = 1, opts = {'caption' : win1}, win = win1)
                #     vis.images(inter_feature[1] * 100, nrow = 1, opts = {'caption' : win2}, win = win2)
                #     vis.images(inter_feature[2] * 100, nrow = 1, opts = {'caption' : win3}, win = win3)
                #     del inter_feature
                #     handle1.remove()
                #     handle2.remove()
                #     handle3.remove()
                
                if (step + 1) % pa['freq'] == 0:
                    if(freqError > 10000000):
                        freqError = -1
                    slideCurveX[freqStep, 0] = step
                    slideCurveX[freqStep, 1] = freqError
                    freqError = 0
                    freqStep += 1
                    if freqStep > 1:
                        vis.scatter(slideCurveX[0:freqStep], torch.ones(freqStep), win = 'win0')
            
            trainError = trainError / step
            if(trainError <= 10000000000):
                curveX[curve, 0] = e
                curveX[curve, 1] = trainError
                curveY[curve] = 1
                curve += 1

            ddml.eval()
            it = valLoader1.__iter__()
            for step, (x1, label1) in enumerate(valLoader1):
                (x2, label2) = it.__next__()
                sim = (label1 == label2)
                x1.requires_grad = False;
                x2.requires_grad = False;
                if(pa['gpu']):
                    x1 = x1.cuda()
                    x2 = x2.cuda()
                    sim = sim.cuda()
                y1 = ddml(x1)
                y2 = ddml(x2)
                y2 = y2.detach()
                loss = lossFun(y1, y2, sim)
                # l2_reg = 0
                # for p in ddml.parameters():
                #     l2_reg += torch.norm(p)
                # loss += pa['lambda'] * l2_reg
                valError = valError + loss.data.item()

            valError = valError / step
            curveX[curve, 0] = e
            curveX[curve, 1] = valError
            curveY[curve] = 2
            curve += 1
            
            if curve > 1:
                vis.scatter(curveX[0:curve], curveY[0:curve], win = 'win1')
            print(e, time.time() - start, trainError, valError)
            torch.save(ddml.state_dict(), workPath + '/featureNetState{}'.format(e))
        return ddml

    def __test__(self, pa, workPath, featureNet, loader1, loader2, tau):
        featureNet.eval()
        if(pa['gpu']):
            featureNet = featureNet.cuda()
        result = torch.zeros(int(loader1.dataset.__len__()), 5)
        idx = 0
        count = 0
        assert loader1.__len__() <= loader2.__len__()
        it1 = loader1.__iter__()
        it2 = loader2.__iter__()
        for step in range(loader1.__len__()):
            (x1, label1) = it1.__next__()
            (x2, label2) = it2.__next__()
            sim = label1 == label2
            sim = sim.float()
            sim = sim * 2 - 1
            if pa['gpu']:
                x1 = x1.cuda()
                x2 = x2.cuda()
                sim = sim.cuda()

            y1 = featureNet(x1)
            y2 = featureNet(x2)
            y = y1 - y2
            y = torch.tensor(y.data, requires_grad = False)
            distance = torch.norm(y, 2, 1)
            pred = distance < tau
            pred = pred.float()
            pred = pred * 2 - 1
            pred = torch.tensor(pred.data, requires_grad = False)

            result[idx : idx + pred.shape[0], 0] = pred
            result[idx : idx + pred.shape[0], 1] = sim
            result[idx : idx + pred.shape[0], 2] = label1
            result[idx : idx + pred.shape[0], 3] = label2
            result[idx : idx + pred.shape[0], 4] = distance

            count += torch.sum(pred == sim)
            idx += pred.shape[0]
        print('accuracy: ', count.item() / idx)
        return result


    def train(self):
        pa = self.pa
        workPath = self.workPath
        trainLoader1 = self.loader[0]
        trainLoader2 = self.loader[1]
        valLoader1 = self.loader[2]
        valLoader2 = self.loader[3]

        featureNet = self.featureNet

        lossFun = DDMLLoss(pa['tau'], pa['beta'])
        if pa['gpu']:
            featureNet = featureNet.cuda()
            lossFun = lossFun.cuda()
        optim = torch.optim.Adam(featureNet.parameters(), lr = pa['lr'], weight_decay = 1e-6)
        
        self.featureNet = self.__train__(pa, workPath, featureNet, trainLoader1, trainLoader2, valLoader1, valLoader2, lossFun, optim)

    def test(self):
        pa = self.pa
        workPath = self.workPath
        featureNet = self.featureNet
        testLoader1 = self.loader[4]
        testLoader2 = self.loader[5]
        result = self.__test__(pa, workPath, featureNet, testLoader1, testLoader2, pa['tau'])
        return result
    
if __name__ == "__main__":
    pass