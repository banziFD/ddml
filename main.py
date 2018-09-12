import torch
from torch.utils.data import DataLoader
from ddml import ResFeature
from ddml import DDMLLoss
from ddml import DDML
import util.data.cifar as cifar
from pretrain import Pretrain

def setPath():
    path = dict()
    # path['datasetPath'] = '/home/spyisflying/dataset/cifar/cifar-10-python'
    # path['workPath'] = '/home/spyisflying/git/ddml/ex'
    path['datasetPath'] = 'd:/dataset/cifar-10-python'
    path['workPath'] = 'd:/git/ddml/ex'
    return path

def setParamPre():
    p = dict()
    p['lr'] = 0.0001
    p['batch'] = 128
    p['epoch'] = 10
    p['gpu'] = True
    p['freq'] = 3
    p['nbClass'] = 10
    return p

def setParam():
    param = dict()
    param['lr'] = 0.00005
    param['batch'] = 2
    param['epoch'] = 5
    param['gpu'] = False
    param['tau'] = 1.5
    param['beta'] = 1
    param['lambda'] = 0.2
    param['freq'] = 3
    param['milestones'] = [30, 60, 90, 120, 150, 180, 210]
    return param

def loaderListPre():
    path = setPath()
    pa = setParamPre()
    workPath = path['workPath']

    trainData = cifar.CifarSet(workPath, 'train', vecLabel = True)
    valData = cifar.CifarSet(workPath, 'val', vecLabel = True)
    testData = cifar.CifarSet(workPath, 'test', vecLabel = True)
    
    trainLoader = DataLoader(trainData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    valLoader = DataLoader(valData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    testLoader = DataLoader(testData, batch_size = pa['batch'], shuffle = True, num_workers = 2)
    
    loaderList = [trainLoader, valLoader, testLoader]
    return loaderList

def loaderList():
    path = setPath()
    pa = setParam()
    
    trainData1 = cifar.CifarSet(path['workPath'], 'train')
    trainData2 = cifar.CifarSet(path['workPath'], 'train')
    valData = cifar.CifarSet(path['workPath'], 'val')
    testData = cifar.CifarSet(path['workPath'], 'test')

    trainLoader1 = DataLoader(trainData1, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    trainLoader2 = DataLoader(trainData2, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    valLoader1 = DataLoader(valData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    valLoader2 = DataLoader(valData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    testLoader1 = DataLoader(testData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    testLoader2 = DataLoader(trainData1, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)

    loaderList = [trainLoader1, trainLoader2, valLoader1, valLoader2, testLoader1, testLoader2]
    return loaderList

def main():
    path = setPath()

    print('Preparing data...')
    # cifar.prepareData(path['datasetPath'], path['workPath'])
    nbClass = 10
    
    print('Loading pretrain data')
    loader = loaderListPre()

    print('Initializing model and pretrain...')
    pa = setParamPre()
    featureNet = ResFeature()
    pre = Pretrain(pa, featureNet, path['workPath'], loader, nbClass)
#     pre.train()
#     torch.save(pre, 'pre')
#     pre.test()
    state = pre.getState()
    
    c = input('Complete pretrain, press y to continue')
    if c != 'y':
        return 0

    print('Loading pretrain information...')
    featureNet.load_state_dict(state)

    print('Loading data...')
    loader = loaderList()

    pa = setParam()
    ddml = DDML(pa, featureNet, path['workPath'], loader)
    print('Training...')
    # ddml.train()
    torch.save(ddml, path['workPath'] + '/ddml')
    
    ddml = torch.load(path['workPath'] + '/ddml')
    print('Testing...')
    result = ddml.test()

if __name__ == '__main__':
    main()