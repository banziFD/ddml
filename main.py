import torch
from torch.utils.data import DataLoader
from ddml import ResFeature
from ddml import DDMLLoss
from ddml import DDML
import util.data.cifar as cifar
import util.data.caltech as caltech
from pretrain import Pretrain

def setPath():
    path = dict()
    path['datasetPath'] = '/home/spyisflying/dataset/caltech20'
    path['workPath'] = '/home/spyisflying/git/ddml/ex'
    return path

def setParamPre():
    p = dict()
    p['lr'] = 0.0001
    p['batch'] = 32
    p['epoch'] = 10
    p['gpu'] = True
    p['freq'] = 3
    p['nbClass'] = 20
    return p

def setParam():
    param = dict()
    param['lr'] = 0.00001
    param['batch'] = 64
    param['epoch'] = 20
    param['gpu'] = True
    param['tau'] = 1.5
    param['beta'] = 2
    param['freq'] = 3
    param['milestones'] = list()
    return param

def loaderListPre():
    path = setPath()
    pa = setParamPre()
    workPath = path['workPath']

    trainData = caltech.CaltechSet(workPath, 'train', vecLabel = True, nbClass = pa['nbClass'])
    valData = caltech.CaltechSet(workPath, 'val', vecLabel = True, nbClass = pa['nbClass'])
    testData = caltech.CaltechSet(workPath, 'test', vecLabel = True, nbClass = pa['nbClass'])
    
    trainLoader = DataLoader(trainData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 4)
    valLoader = DataLoader(valData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    testLoader = DataLoader(testData, batch_size = pa['batch'], shuffle = True, num_workers = 2)
    
    loaderList = [trainLoader, valLoader, testLoader]
    return loaderList

def loaderList():
    path = setPath()
    pa = setParam()
    
    trainData1 = caltech.CaltechSet(path['workPath'], 'train', vecLabel = False, nbClass = 20)
    trainData2 = caltech.CaltechSet(path['workPath'], 'train', vecLabel = False, nbClass = 20)
    valData = caltech.CaltechSet(path['workPath'], 'val', vecLabel = False, nbClass = 20)
    testData = caltech.CaltechSet(path['workPath'], 'test', vecLabel = False, nbClass = 20)

    trainLoader1 = DataLoader(trainData1, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    trainLoader2 = DataLoader(trainData2, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    valLoader1 = DataLoader(valData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    valLoader2 = DataLoader(trainData1, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    testLoader1 = DataLoader(testData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    testLoader2 = DataLoader(testData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)

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
    
#     pre.train()
    pre.classifier.load_state_dict(torch.load(path['workPath'] + '/pretrainState'))
    pre.test()
    torch.save(pre, path['workPath'] + '/pretrain')
    
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
#     ddml.featureNet.load_state_dict(torch.load(path['workPath'] + '/featureNetState'))
    print('Testing...')
    for i in range(pa['epoch']):
        ddml.featureNet.load_state_dict(torch.load(path['workPath'] + '/featureNetState{}'.format(i)))
        result = ddml.test()
        torch.save(result, path['workPath'] + '/result{}'.format(i))
    torch.save(ddml, path['workPath'] + '/ddml')

if __name__ == '__main__':
    main()