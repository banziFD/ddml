import os
from glob import glob
import json
import random
from PIL import Image
import numpy as np
import torch
import torchvision.transforms as transforms

def yaleBData(datasetPath, workPath, select):
    folders = [str(i).zfill(2) for i in select]
    l = 0
    label = list()
    idx = 0
    if not os.path.isdir(workPath + '/image'):
        os.mkdir(workPath + '/image')

    for folder in folders:
        files = glob(datasetPath + '/yaleB{}/*.pgm'.format(folder))
        for f in files:
            img = Image.open(f)
            label.append(l)
            img.save(workPath + '/image/{}.pgm'.format(idx))
            idx += 1
        l += 1
    
    test = np.random.choice(len(label), len(label) // 10, replace = False).tolist()
    train = list()
    for i in range(len(label)):
        if i not in test:
            train.append(i)
    random.shuffle(train)
    json.dump(label, open(workPath + '/label.json', 'w'))
    json.dump(train, open(workPath + '/train.json', 'w'))
    json.dump(test, open(workPath + '/test.json', 'w'))
    json.dump(test, open(workPath + '/val.json', 'w'))


def prepareData(datasetPath, workPath, select = list(range(1, 11))):
    yaleBData(datasetPath, workPath, select)

class YaleBSet(torch.utils.data.Dataset):
    def __init__(self, workPath, mode = 'train', vecLabel = False, nbClass = 10):
        super(YaleBSet, self).__init__()
        self.mode = mode
        self.image = workPath + '/image'
        self.label = json.load(open(workPath + '/label.json'))
        self.key = json.load(open(workPath + '/{}.json'.format(self.mode)))
        self.transform = transforms.Compose([transforms.Grayscale(3), 
        transforms.Resize([224, 224]), transforms.ToTensor(), transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        self.vecLabel = vecLabel
        self.nbClass = nbClass

    def __getitem__(self, index):
        k = self.key[index]
        x = Image.open(self.image + '/{}.pgm'.format(k))
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
    datasetPath = 'D:/dataset/CroppedYale'
    workPath = 'D:/dataset/CroppedYale'
    prepareData(datasetPath, workPath, select = list(range(1, 10)))

