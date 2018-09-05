import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from visdom import Visdom
import time
import utils_ddml
import utils_data
from cifar.utils_model import CNN as Feature

def make_hook(flag, data):
    if(flag == 'f'):
        def hook(m, input, output):
            data.append(input)
        return hook
    if(flag == 'b'):
        def hook(m, input, output):
            data.append(output)
        return hook

def getFeature(state):
    feature = Feature()
    feature.load_state_dict(state)
    return feature.getFeatureDict()

def train(pa, workPath, ddml, trainLoader1, trainLoader2, valLoader1, valLoader2, lossFun, optim):
    vis = Visdom()
    curveX = torch.zeros(pa['epoch'] * 2, 2)
    curveY = torch.zeros(pa['epoch'] * 2)
    curve = 0
    
    for e in range(pa['epoch']):
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
#             if(step == 0):
#                 inter_feature = []
                # handle1 = ddml.feature.linear1.register_forward_hook(make_hook('f', inter_feature))
                # handle2 = ddml.feature.linear2.register_forward_hook(make_hook('f', inter_feature))
                # handle3 = ddml.feature.linear3.register_forward_hook(make_hook('f', inter_feature))
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
#                 inter_feature = [item[0][0:10] for item in inter_feature]
                # temp = inter_feature[5]
                # inter_feature = inter_feature[0:3]
                # inter_feature[2] = inter_feature[2] - temp
                # inter_feature[0] = inter_feature[0].reshape(-1, 3, 32, 32)
                # inter_feature[1] = inter_feature[1].reshape(-1, 1, 16, 16)
                # inter_feature[2] = inter_feature[2].reshape(-1, 1, 16, 16)
                # vis.images(inter_feature[0], nrow = 1, opts = {'caption' : win1}, win = win1)
                # vis.images(inter_feature[1] * 100, nrow = 1, opts = {'caption' : win2}, win = win2)
                # vis.images(inter_feature[2] * 100, nrow = 1, opts = {'caption' : win3}, win = win3)
                # del inter_feature
                # handle1.remove()
                # handle2.remove()
                # handle3.remove()
            
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
        
        vis.scatter(curveX[0:curve], curveY[0:curve], win = 'win1')
        print(e, time.time() - start, trainError, valError)
        
        torch.save(ddml.state_dict(), workPath + '/ddmlState{}'.format(e))
    torch.save(ddml, workPath + '/ddml')

def test(pa, workPath, ddml, loader1, loader2, tau):
    ddml.eval()
    if(pa['gpu']):
        ddml = ddml.cuda()
    result = torch.zeros(int(loader1.dataset.__len__()), 5)
    idx = 0
    count = 0
    assert loader1.__len__() <= loader2.__len__()
    it1 = loader1.__iter__()
    it2 = loader2.__iter__()
    for step in range(loader1.__len__()):
        print(step)
        (x1, label1) = it1.__next__()
        (x2, label2) = it2.__next__()
        sim = label1 == label2
        sim = sim.float()
        sim = sim * 2 - 1
        if pa['gpu']:
           x1 = x1.cuda()
           x2 = x2.cuda()
           sim = sim.cuda()

        y1 = ddml(x1)
        y2 = ddml(x2)
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


def main():
    path = setPath()
    pa = setParam()

    print('Preparing data...')
#     utils_data.prepareData(path['datasetPath'], path['workPath'])
    
    state = torch.load(path['workPath'] + '/modelState')
    feature = getFeature(state)

    print('Initializing model and loading data...')
    ddml = utils_ddml.DDML(feature)
    lossFun = utils_ddml.DDMLLoss(pa['tau'], pa['beta'])
    if pa['gpu']:
        ddml = ddml.cuda()
        lossFun = lossFun.cuda()
    optim = torch.optim.Adam(ddml.parameters(), lr = pa['lr'], weight_decay = 1e-6)

    trainData1 = utils_data.DDMLData(path['workPath'], 'train')
    trainData2 = utils_data.DDMLData(path['workPath'], 'train')
    valData = utils_data.DDMLData(path['workPath'], 'val')
    testData = utils_data.DDMLData(path['workPath'], 'test')


    trainLoader1 = DataLoader(trainData1, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    trainLoader2 = DataLoader(trainData2, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    valLoader1 = DataLoader(valData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    valLoader2 = DataLoader(valData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    testLoader1 = DataLoader(testData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    testLoader2 = DataLoader(trainData1, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)

    print('Training...')
#     train(pa, path['workPath'], ddml, trainLoader1, trainLoader2, valLoader1, valLoader2, lossFun, optim)
#     torch.save(ddml, path['workPath'] + '/ddml')

    ddml = torch.load(path['workPath'] + '/ddml')

    print('Testing...')
    result = test(pa, path['workPath'], ddml, testLoader1, testLoader2, pa['tau'])
    torch.save(result, path['workPath'] + '/result')
    
    
def setParam():
    param = dict()
    param['lr'] = 0.00005
    param['batch'] = 128
    param['epoch'] = 5
    param['gpu'] = True
    param['tau'] = 1.5
    param['beta'] = 1
    param['lambda'] = 0.2
    param['freq'] = 3
    param['milestones'] = [30, 60, 90, 120, 150, 180, 210]
    return param

def setPath():
    path = dict()
    path['datasetPath'] = '/home/spyisflying/dataset/cifar/cifar-10-python'
    path['workPath'] = '/home/spyisflying/git/ddml/ex'
#     path['datasetPath'] = 'd:/dataset/cifar-10-python'
#     path['workPath'] = 'd:/git/ddml/ex'
    return path

if __name__ == '__main__':
    main()