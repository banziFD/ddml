import torch
from torch.utils.data import DataLoader
from ddml import ResFeature
from ddml import DDMLLoss
from ddml import DDML
import util.data.cifar as cifar
from pretrain import Pretrain

def setPath():
    path = dict()
    path['datasetPath'] = '/home/spyisflying/dataset/cifar/cifar-100-python'
    path['workPath'] = '/home/spyisflying/git/ddml/ex'
#     path['datasetPath'] = 'd:/dataset/cifar-10-python'
#     path['workPath'] = 'd:/git/ddml/ex'
    return path

def setParamPre():
    p = dict()
    p['lr'] = 0.00001
    p['batch'] = 128
    p['epoch'] = 5
    p['gpu'] = True
    p['freq'] = 3
    p['nbClass'] = 10
    return p

def setParam():
    param = dict()
    param['lr'] = 0.00005
    param['batch'] = 128
    param['epoch'] = 8
    param['gpu'] = True
    param['tau'] = 1.5
    param['beta'] = 1
    param['lambda'] = 0.2
    param['freq'] = 3
    param['milestones'] = [8, 13, 17]
    return param

def loaderListPre():
    path = setPath()
    pa = setParamPre()
    workPath = path['workPath']

    trainData = cifar.CifarSet(workPath, 'train', vecLabel = True, nbClass = 100)
    valData = cifar.CifarSet(workPath, 'val', vecLabel = True, nbClass = 100)
    testData = cifar.CifarSet(workPath, 'test', vecLabel = True, nbClass = 100)
    
    trainLoader = DataLoader(trainData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    valLoader = DataLoader(valData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    testLoader = DataLoader(testData, batch_size = pa['batch'], shuffle = True, num_workers = 2)
    
    loaderList = [trainLoader, valLoader, testLoader]
    return loaderList

def loaderList():
    path = setPath()
    pa = setParam()
    
    trainData1 = cifar.CifarSet(path['workPath'], 'train', vecLabel = False)
    trainData2 = cifar.CifarSet(path['workPath'], 'train', vecLabel = False)
    valData = cifar.CifarSet(path['workPath'], 'val', vecLabel = False)
    testData = cifar.CifarSet(path['workPath'], 'test', vecLabel = False)

    trainLoader1 = DataLoader(trainData1, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    trainLoader2 = DataLoader(trainData2, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    valLoader1 = DataLoader(valData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    valLoader2 = DataLoader(trainData1, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    testLoader1 = DataLoader(testData, batch_size = pa['batch'], shuffle = False, drop_last = True, num_workers = 2)
    testLoader2 = DataLoader(trainData1, batch_size = pa['batch'], shuffle = False, drop_last = True, num_workers = 2)

    loaderList = [trainLoader1, trainLoader2, valLoader1, valLoader2, testLoader1, testLoader2]
    return loaderList

def main():
    path = setPath()

    print('Preparing data...')
    cifar.prepareData100(path['datasetPath'], path['workPath'])
    nbClass = 100
    
    print('Loading pretrain data')
    loader = loaderListPre()

    print('Initializing model and pretrain...')
    pa = setParamPre()
    featureNet = ResFeature()
    pre = Pretrain(pa, featureNet, path['workPath'], loader, nbClass)
    
    pre.train()
    pre.classifier.load_state_dict(torch.load(path['workPath'] + '/pretrainState'))
    pre.test()
    torch.save(pre, 'pretrain')
    
#     c = input('Complete pretrain, press y to continue /n')
#     if c != 'y':
#         return 0

    print('Loading pretrain information...')
    state = pre.getState()
    featureNet.load_state_dict(state)

    print('Loading data...')
    loader = loaderList()

    pa = setParam()
    ddml = DDML(pa, featureNet, path['workPath'], loader)
    print('Training...')
    ddml.train()
    ddml.featureNet.load_state_dict(torch.load(path['workPath'] + '/featureNetState'))
    print('Testing...')
    result = ddml.test()
    torch.save(result, path['workPath'] + '/result')


if __name__ == '__main__':
    main()