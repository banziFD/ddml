import torch
import torch.nn as nn
from torch.utils.data import DataLoader 
import time
from visdom import Visdom
from ddml import ResFeature

class Classifier(nn.Module):
    def __init__(self, state_dict, nbClass):
        super(Classifier, self).__init__()
        self.feature = ResFeature()
        self.feature.load_state_dict(state_dict)
        self.linear = nn.Linear(self.feature.length, nbClass)
    
    def forward(self, x):
        y = self.feature(x)
        y = self.linear(y)
        return y

    def getState(self):
        return self.feature.state_dict()

class Pretrain:
    def __init__(self, pa, featureNet, workPath, loaderList, nbClass):
        self.classifier = Classifier(featureNet.state_dict(), nbClass)
        self.workPath = workPath
        self.pa = pa
        self.loaderList = loaderList
        assert loaderList[0].dataset.vecLabel == True
   

    def __makeHook__(flag, data):
        if flag == 'f':
            def hook(m, input, output):
                data.append(input)
            return hook
        if flag == 'b':
            def hook(m, input, output):
                data.append(output)
            return hook
    
    def __train__(self, pa, workPath, model, trainLoader, valLoader, lossFun, optim):
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
            for step, (x, y, k) in enumerate(trainLoader):
                # if step == 0:
                #     interFeature = []
                #     handle1 = model.conv1.register_forward_hook(make_hook('f', interFeature))
                #     handle2 = model.layer1.register_forward_hook(make_hook('f', interFeature))

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

                # if step == 0:
                #     handle1.remove()
                #     handle2.remove()
                    
                if (step + 1) % pa['freq'] == 0:
                    slideCurveX[freqStep, 0] = step
                    slideCurveX[freqStep, 1] = freqError
                    freqError = 0
                    freqStep += 1
                    if freqStep > 1:
                        vis.scatter(slideCurveX[0:freqStep], torch.ones(freqStep), win = 'win0')
                        
            trainError = trainError / (step + 1)
            curveX[curve, 0] = e
            curveX[curve, 1] = trainError
            curveY[curve] = 1
            curve += 1

            model.eval()
            for step, (x, y, k) in enumerate(valLoader):
                if(pa['gpu']):
                    x = x.cuda()
                    y = y.cuda()
                
                optim.zero_grad()
                pred = model(x)
                loss = lossFun(pred, y)
                valError += loss.data.item()
            
            valError = valError / (step + 1)
            curveX[curve, 0] = e
            curveX[curve, 1] = valError
            curveY[curve] = 2
            curve += 1 
            vis.scatter(curveX[0:curve], curveY[0:curve], win = 'win1')

            print(e, time.time() - start, trainError, valError)
            torch.save(model.state_dict(), workPath + '/pretrainState{}'.format(e))
        
        return model

    def train(self):
        pa = self.pa
        workPath = self.workPath
        print('Initializing pretrain model and loading data...')
        classifier = self.classifier
        lossFun = nn.MSELoss(size_average = True)

        if self.pa['gpu']:
            classifier = classifier.cuda()
            lossFun = lossFun.cuda()

        optim = torch.optim.Adam(classifier.parameters(), lr = pa['lr'], weight_decay = 1e-6)

        trainLoader = self.loaderList[0]
        valLoader = self.loaderList[1]

        print('Pretraining...')
        self.classifier = self.__train__(pa, workPath, classifier, trainLoader, valLoader, lossFun, optim)
    
    def test(self):
        pa = self.pa
        workPath = self.workPath
        classifier = self.classifier
        testLoader = self.loaderList[2]

        print('Classification testing...')
        count = 0
        t = 0
        if pa['gpu']:
            classifier = classifier.cuda()
        
        classifier = classifier.eval()
        for step, (x, y, k) in enumerate(testLoader):
            if pa['gpu']:
                x = x.cuda()
                y = y.cuda()
            
            lbl = y.nonzero()[:, 1]
            pred = classifier(x)

            pred = torch.argmax(pred, dim = 1)
            res = (pred == lbl)

            count += res.shape[0]
            t += torch.sum(res).data.item()
        print(t, count, t / count)

    def getState(self):
        return self.classifier.getState()
    
if __name__ == '__main__':
    pass