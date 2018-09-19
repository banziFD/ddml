import torch
from torch.utils.data import DataLoader
from ddml import ResFeature
from ddml import DDMLLoss
from ddml import DDML
import util.data.cifar as cifar
import util.data.caltech as caltech
import util.data.msrcv as msrcv
import json

def latentFeature(featureNet, loader):
    size = loader.dataset.__len__()

    label = torch.zeros(size, dtype = torch.uint8)
    feature = torch.zeros(size, 512)
    key = torch.zeros(size, dtype = torch.long)

    featureNet.eval()
    featureNet = featureNet.cuda()

    for step, (x, l, k) in enumerate(loader):
        if pa['gpu']:
            x = x.cuda()
        label[step] = l[0]
        feature[step] = featureNet(x)[0].data()
        key[step] = k
    
    label = label.numpy()
    feature = feature.numpy()
    key = key.numpy()
    return label, feature, key

def getLoader(workPath):
    data = cifar.CifarSet(workPath, 'train', vecLabel = False)
    loader = DataLoader(trainData, batch_size = 1, shuffle = False, drop_last = False, num_workers = 2)
    return loader

if __name__ == '__main__':
    workPath = '/home/spyisflying/git/ddml/ex'

    print('Pretraing data...')
    cifar.prepareData10(path['datasetPath'], path['workPath'])

    featureNet = ResFeature()
    featureNet.load_state_dict(torch.load(path['workPath'] + '/featureNetState'))

    loader = getLoader()

    print('Extracting features...')
    label, feature, key = latentFeature(featureNet, loader)

    print('Saving data as numpy array')
    np.save(label, 'label')
    np.save(feature, 'feature')
    np.save(key, 'key')