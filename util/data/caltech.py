import pickle
import numpy as np 
import json
import random
from glob import glob
from PIL import Image  
from scipy.io import loadmat
import re
import torch
import torch
import torchvision.transforms as transforms

def caltechData(datasetPath, outputPath):
    classes = glob(datasetPath)
    classid = 0
    meta = dict()
    for cl in classes:
        meta[cl.split('\\')[1]] = classid
        classid += 1
    
    file = open('meta.txt', 'w')
    for key in meta.keys():
        file.write(key)
        file.write('  ')
        file.write(str(meta[key]))
        file.write('\n')
    file.close()
    print(meta)

    idx = 0
    label = list()
    for key in meta.keys():
        image = glob('7/{}/*.jpg'.format(key))
        for i in image:
            name = re.findall('[0-9]{4}', i)[0]
            img = Image.open(i)
            ann = loadmat('./Annotations/{}/annotation_{}'.format(key, name))
            bbox = ann['box_coord'][0]
            img = img.crop((bbox[2], bbox[0], bbox[3], bbox[1]))
            label.append(meta[key])
            img.save(outputPath + '{}.jpg'.format(idx))
            idx += 1
    torch.save('label', torch.tensor(label))
    
def prepareData(datasetPath, workPath):
    label = torch.load(datasetPath + '/label')
    test = np.random.choice(label.shape[0], (label.shape[0] // 10,))
    test = test.tolist()
    train = list()
    for i in range(label.shape[0]):
        if i not in test:
            train.append(i)
    random.shuffle(train)
    json.dump(train, open(workPath + '/train.json', 'w'))
    json.dump(test, open(workPath + '/test.json', 'w'))
    json.dump(test, open(workPath + '/val.json', 'w'))

class CaltechSet(torch.utils.data.Dataset):
    def __init__(self, workPath, mode = 'train', vecLabel = False, nbClass = 7):
        super(CaltechSet, self).__init__()
        self.mode = mode
        self.image = workPath + '/caltech7/'
        self.label = torch.load(workPath + '/caltech7/label')
        self.key = json.load(open(workPath + '/caltech7/{}.json'.format(mode)))
        self.vecLabel = vecLabel
        self.nbClass = nbClass
        self.transform = transforms.Compose(
            [transforms.Grayscale(3),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])
        
    def __getitem__(self, index):
        k = self.key[index]
        x = self.image + '{}.jpg'.format(k)
        x = Image.open(x)
        x = self.transform(x)
        y = self.label[k]
        if(self.vecLabel):
            t = torch.zeros(self.nbClass)
            t[y] = 1
            y = t
        return x, y

    def __len__(self):
        return len(self.key)

if __name__ == '__main__':
    pass