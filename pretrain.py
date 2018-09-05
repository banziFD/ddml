import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
import time
from visdom import Visdom
import utils_model
import utils_data

def make_hook(flag, data):
    if flag == 'f':
        def hook(m, input, output):
            data.append(input)
        return hook
    if flag == 'b':
        def hook(m, input, output):
            data.append(output)
        return hook

def train(pa, workPath, model, trainLoader, valLoader, lossFun,optim):
    vis = Visdom()
    curveX = torch.zeros(pa['epoch'] * 2, 2)
    curveY = torch.zeros(pa['epoch'] * 2)
    curve = 0
    
    for e in range(pa['epoch']):
        start = time.time()
        trainError = 0
        valError = 0
        
        slideCurveX = torch.zeros(len(trainLoader), 2)
        freqError = 0
        freqStep = 0
        slideCurveY = torch.zeros(len(trainLoader))

        model.train()
        for step, (x, y) in enumerate(trainLoader):
            
#             if step == 0:
#                 interFeature = []
#                 handle1 = model.conv1.register_forward_hook(make_hook('f', interFeature))
#                 handle2 = model.layer1.register_forward_hook(make_hook('f', interFeature))

            if pa['gpu']:
                x = x.cuda()
                y = y.cuda()
            
            pred = model(x)
            loss = lossFun(pred, y)
            optim.zero_grad()
            loss.backward()
            optim.step()

            trainError += loss.data.item()
            freqError += loss.data.item()

#             if step == 0:
#                 handle1.remove()
#                 handle2.remove()
                
            if (step + 1) % pa['freq'] == 0:
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

        model.eval()
        for step, (x, y) in enumerate(valLoader):
            if(pa['gpu']):
                x = x.cuda()
                y = y.cuda()
            
            optim.zero_grad()
            pred = model(x)
            loss = lossFun(pred, y)
            valError += loss.data.item()
        
        valError = valError / step
        curveX[curve, 0] = e
        curveX[curve, 1] = valError
        curveY[curve] = 2
        curve += 1 

        vis.scatter(curveX[0:curve], curveY[0:curve], win = 'win1')
        print(e, time.time() - start, trainError, valError)

        torch.save(model.state_dict(), workPath + '/modelState{}'.format(e))
    
    torch.save(model, workPath + '/cnn')

def test(pa, model, testLoader):
    count = 0
    t = 0
    if pa['gpu']:
        model = model.cuda()
    model = model.eval()

    for step, (x, y) in enumerate(testLoader):
        if pa['gpu']:
            x = x.cuda()
            y = y.cuda()
        
        lbl = y.nonzero()[:, 1]
        pred = model(x)

        pred = torch.argmax(pred, dim = 1)
        res = (pred == lbl)

        count += res.shape[0]
        t += torch.sum(res).data.item()

    print(t, count, t / count)

def setPath():
    path = dict()
    path['datasetPath'] = '/home/spyisflying/dataset/cifar/cifar-10-python'
    path['workPath'] = '/home/spyisflying/git/ddml/cifarTest/ex'
#     path['datasetPath'] = 'd:/dataset/cifar-10-python'
#     path['workPath'] = 'd:/git/ddml/ex'
    return path

def setParam():
    p = dict()
    p['lr'] = 0.0001
    p['batch'] = 128
    p['epoch'] = 10
    p['gpu'] = True
    p['freq'] = 3
    return p

if __name__ == '__main__':
    path = setPath()
    pa = setParam()
    
    print('Preparing data...')
#     utils_data.prepareData(path['datasetPath'], path['workPath'])
    
    print('Initializing model and loading data...')
    cnn = utils_model.CNN(pa['nbClass'])
    cnn = torch.load(path['workPath'] + '/cnn')
    cnn.load_state_dict(torch.load(path['workPath'] + '/modelState7'))
    lossFun = nn.MSELoss(size_average = True)

    if pa['gpu']:
        cnn = cnn.cuda()
        lossFun = lossFun.cuda()
    optim = torch.optim.Adam(cnn.parameters(), lr = pa['lr'], weight_decay = 1e-6)

    trainData = utils_data.Dataset(path['workPath'], 'train')
    valData = utils_data.Dataset(path['workPath'], 'val')
    testData = utils_data.Dataset(path['workPath'], 'test')
    
#     trainLoader = DataLoader(trainData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    valLoader = DataLoader(valData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    testLoader = DataLoader(testData, batch_size = pa['batch'], shuffle = True, num_workers = 2)
#     testLoader = DataLoader(trainData, batch_size = pa['batch'], shuffle = True)
    print('Training...')
#     train(pa, path['workPath'], cnn, trainLoader, valLoader, lossFun, optim)

#     cnn = torch.load(path['workPath'] + '/cnn')
    
    print('Testing...')
    test(pa, cnn, testLoader)
    