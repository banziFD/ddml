import os
from glob import glob
import json
import random
from PIL import Image
import numpy as np 
import torch
import torchvision
import torchvision.transforms as transforms

def coilData(datasetPath, workPath, nbClass):
    label = list()
    l = 0
    idx = 0
    if not os.path.isdir(workPath + '/image'):
        os.mkdir(workPath + '/image')
    
    for c in range(1, nbClass + 1):
        files = glob(datasetPath + '/obj{}_*.png'.format(c))
        for f in files:
            img = Image.open(f)
            img.save(workPath + '/image/{}.png'.format(idx))
            label.append(l)
            idx += 1
        l += 1

    test = np.random.choice(len(label), len(label) // 10, replace = False).tolist()
    train = list()
    for i in range(len(label)):
        if i not in test:
            train.append(i)
    random.shuffle(train)
    json.dump(train, open(workPath + '/train.json', 'w'))
    json.dump(test, open(workPath + '/test.json', 'w'))
    json.dump(test, open(workPath + '/val.json', 'w')) 
    json.dump(label, open(workPath + '/label.json', 'w'))

def prepareData20(datasetPath, workPath):
    coilData(datasetPath, workPath, 20)

def prepareData100(datasetPath, workPath):
    coilData(datasetPath, workPath, 100)

class CoilSet(torch.utils.data.Dataset):
    def __init__(self, workPath, mode = 'train', vecLabel = 'False', nbClass = 40):
        super(CoilSet, self).__init__()
        self.mode = mode
        self.image = workPath + '/image'
        self.label = json.load(open(workPath + '/label.json'))
        self.key = json.load(open(workPath + '/{}.json'.format(mode)))
        self.transform = transforms.Compose(
            [transforms.Grayscale(3),
            transforms.Resize([224, 224]), 
            transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        self.vecLabel = vecLabel
        self.nbClass = nbClass
    
    def __getitem__(self, index):
        k = self.key[index]
        x = Image.open(self.image + '/{}.png'.format(k))
        x = self.transform(x)
        y = self.label[k]
        if self.vecLabel:
            t = torch.zeros(self.nbClass)
            t[y] = 1
            y = t
        return x, y, k
    
    def __len__(self):
        return len(self.key)

if __name__ == '__main__':
    datasetPath = 'D:/dataset/coil-100'
    workPath = 'D:/dataset/coil-100'
    prepareData100(datasetPath, workPath)