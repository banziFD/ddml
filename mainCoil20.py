import torch
from torch.utils.data import DataLoader
from ddml import ResFeature
from ddml import DDMLLoss
from ddml import DDML
import util.data.cifar as cifar
import util.data.caltech as caltech
import util.data.msrcv as msrcv
import util.data.yaleB as yaleB
import util.data.orl as orl
import util.data.coil as coil
from pretrain import Pretrain

def setPath():
    path = dict()
    path['datasetPath'] = ''
    path['workPath'] = '/home/spyisflying/git/ddml/ex/coil20'
    return path

def setParamPre():
    p = dict()
    p['lr'] = 0.0001
    p['batch'] = 32
    p['epoch'] = 15
    p['gpu'] = True
    p['freq'] = 3
    p['nbClass'] = 20
    return p

def setParam():
    param = dict()
    param['lr'] = 0.00005
    param['batch'] = 128
    param['epoch'] = 70
    param['gpu'] = True
    param['tau'] = 1.5
    param['beta'] = 1
    param['freq'] = 1
    param['milestones'] = list()
    return param

def loaderListPre():
    path = setPath()
    pa = setParamPre()
    workPath = path['workPath']

    trainData = coil.CoilSet(workPath, 'train', vecLabel = True, nbClass = pa['nbClass'])
    valData = coil.CoilSet(workPath, 'val', vecLabel = True, nbClass = pa['nbClass'])
    testData = coil.CoilSet(workPath, 'test', vecLabel = True, nbClass = pa['nbClass'])
    
    trainLoader = DataLoader(trainData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 4)
    valLoader = DataLoader(valData, batch_size = pa['batch'], shuffle = True, num_workers = 2)
    testLoader = DataLoader(testData, batch_size = pa['batch'], shuffle = True, num_workers = 2)
    
    loaderList = [trainLoader, valLoader, testLoader]
    return loaderList

def loaderList():
    path = setPath()
    pa = setParam()
    
    trainData1 = coil.CoilSet(path['workPath'], 'train', vecLabel = False)
    trainData2 = coil.CoilSet(path['workPath'], 'train', vecLabel = False)
    valData1 = coil.CoilSet(path['workPath'], 'val', vecLabel = False)
    valData2 = coil.CoilSet(path['workPath'], 'val', vecLabel = False)
    testData1 = coil.CoilSet(path['workPath'], 'test', vecLabel = False)
    testData2 = coil.CoilSet(path['workPath'], 'test', vecLabel = False)

    trainLoader1 = DataLoader(trainData1, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    trainLoader2 = DataLoader(trainData2, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    valLoader1 = DataLoader(valData1, batch_size = 8, shuffle = True, drop_last = True, num_workers = 2)
    valLoader2 = DataLoader(valData2, batch_size = 8, shuffle = True, drop_last = True, num_workers = 2)
    testLoader1 = DataLoader(testData1, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    testLoader2 = DataLoader(testData2, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)

    loaderList = [trainLoader1, trainLoader2, valLoader1, valLoader2, testLoader1, testLoader2]
    return loaderList

def main():
    path = setPath()

    print('Preparing data...')
    
    print('Loading pretrain data')
    loader = loaderListPre()

    print('Initializing model and pretrain...')
    pa = setParamPre()
    nbClass = pa['nbClass']
    featureNet = ResFeature()
    pre = Pretrain(pa, featureNet, path['workPath'], loader, nbClass)
    
    pre.train()
    pre.classifier.load_state_dict(torch.load(path['workPath'] + '/pretrainState'))
    pre.test()
    torch.save(pre, path['workPath'] + '/pretrain')
    
    c = input('Complete pretrain, press y to continue /n')
    if c != 'y':
        return 0

    print('Loading pretrain information...')
    state = pre.getState()
    featureNet.load_state_dict(state)

    print('Loading data...')
    loader = loaderList()

    pa = setParam()
    ddml = DDML(pa, featureNet, path['workPath'], loader)
    print('Training...')
#     ddml.train()
    ddml.featureNet.load_state_dict(torch.load(path['workPath'] + '/featureNetState'))
    print('Testing...')
    result = ddml.test()
    torch.save(result, path['workPath'] + '/result')
    torch.save(ddml, path['workPath'] + '/ddml')

if __name__ == '__main__':
    main()