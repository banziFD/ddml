import pickle
import torch
from torch.utils.data import DataLoader
from ddml import ResFeature
from ddml import DDMLLoss
from ddml import DDML
import util.data.cifar as cifar
from pretrain import Pretrain  

def latentFeature(pa, workPath, featureNet, loader:
    data = dict()
    size = loader.dataset.__len__()
    data['image'] = list()
    data['label'] = torch.zeros(size, dtype = torch.uint8)
    data['feature'] = torch.zeros(size, 512)

    featureNet.eval()
    if pa['gpu']:
        featureNet = featureNet.cuda()

    size = loader.dataset().shape[0]
    feature = torch.zeros(size, 512)
    for step, (x, label, img) in enumerate(loader):
        if pa['gpu']:
            x = x.cuda()
        feature = 


    return data

def setPath():
    path = dict()
    path['datasetPath'] = '/home/spyisflying/dataset/cifar/cifar-10-python'
    path['workPath'] = '/home/spyisflying/git/ddml/ex'
#     path['datasetPath'] = 'd:/dataset/cifar-10-python'
#     path['workPath'] = 'd:/git/ddml/ex'
    return path

def setParam():
    param = dict()
    param['batch'] = 128
    param['gpu'] = True
    return param

def loaderList():
    path = setPath()
    pa = setParam()
    
    trainData = cifar.CifarSet(path['workPath'], 'train', vecLabel = False)
    testData = cifar.CifarSet(path['workPath'], 'test', vecLabel = False)

    trainLoader = DataLoader(trainData1, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)
    testLoader = DataLoader(testData, batch_size = pa['batch'], shuffle = True, drop_last = True, num_workers = 2)

    loaderList = [trainLoader, testLoader]
    return loaderList

if __name__ == '__main__':
    pa = setParam()
    path = setPath()

    print('Pretraing data...')
    cifar.prepareData10(path['datasetPath'], path['workPath'])

    featureNet = ResFeature()
    featureNet.load_state_dict(torch.load(path['workPath'] + '/featureNetState'))

    dataTrain = latentFeature(pa, path['workPath'], featureNet, loaderList[0])
    dataTest = latentFeature(pa, path['workPath'], featureNet, loaderList[1])

    pickle.dump(dataTrain, open('cifar10Train', 'w'))
    pickle.dump(dataTest, oepn('cifar10Test', 'w'))
