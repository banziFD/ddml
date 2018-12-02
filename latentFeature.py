import torch
from torch.utils.data import DataLoader
from ddml import ResFeature
from ddml import DDMLLoss
from ddml import DDML
import util.data.cifar as cifar
import util.data.caltech as caltech
import util.data.msrcv as msrcv
import numpy as np

def latentFeature(featureNet, loader):
    size = loader.dataset.__len__()

    label = torch.zeros(size, dtype = torch.uint8)
    feature = torch.zeros(size, 512)
    key = torch.zeros(size, dtype = torch.long)

    featureNet.eval()
    featureNet = featureNet.cuda()

    for step, (x, l, k) in enumerate(loader):
        x = x.cuda()
        label[step] = l[0]
        current = featureNet(x)
        current = current.data
        feature[step] = current
        key[step] = k
    
    label = label.numpy()
    feature = feature.numpy()
    key = key.numpy()
    return label, feature, key

def getLoader(workPath):
    data = caltech.CaltechSet(workPath, 'image_list', vecLabel = False, nbClass = 10)
    loader = DataLoader(data, batch_size = 128, shuffle = False, drop_last = False, num_workers = 4)
    return loader

if __name__ == '__main__':
    workPath = '/home/spyisflying/git/ddml/ex'

    print('Preparing data...')
#     cifar.prepareData10('/home/spyisflying/dataset/cifar/cifar-10-python', workPath)

    featureNet = ResFeature()
    # featureNet.load_state_dict(torch.load(workPath + '/featureNetState'))

    loader = getLoader(workPath)

    print('Extracting features...')
    label, feature, key = latentFeature(featureNet, loader)

    print('Saving data as numpy array')
    
    mode = 'caltech7'
    np.save(mode + 'label', label)
    np.save(mode + 'feature', feature)
    np.save(mode + 'key', key)