import pickle
import numpy as np 
import glob
import torch
import torch
import torchvision.transforms as transforms

def cifarUnpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding = 'bytes')
        return dict

def cifar10Data(datasetPath, vecLabel, select = None):
    trainFile = glob.glob(datasetPath + '/data*')
    trainLabel = []
    trainImage = []
    testFile = glob.glob(datasetPath + '/test*')

    for f in train_files:
        current = cifarUnpickle(f)
        currentLabel = current[b'labels']
        currentLabel = torch.tensor(currentLabel)
        currentImage = current[b'data']
        currentImage = torch.from_numpy(currentImage)
        trainLabel.append(currentLabel)
        trainImage.append(currentImage)
    trainLabel = tuple(trainLabel)
    trainImage = tuple(trainImage)
    trainLabel = torch.cat(trainLabel, dim = 0)
    trainImage = torch.cat(trainImage, dim = 0)

    testFile = cifar_unpickle(testFile[0])
    testLabel = torch.tensor(testFile[b'labels'])
    testImage = torch.from_numpy(testLabel[b'data'])

    trainImage = trainImage.reshape(trainImage.shape[0], 3, 32, 32)
    testImage = testImage.reshape(testImage.shape[0], 3, 32, 32)
    trainImage = trainImage.permute(0, 2, 3, 1).numpy()
    testImage = testImage.permute(0, 2, 3, 1).numpy()

    if vecLabel:
        train_label_temp = torch.zeros(train_labels.shape[0], 10, dtype = torch.float)
        test_label_temp = torch.zeros(test_labels.shape[0], 10, dtype = torch.float)
        for i in range(train_images.shape[0]):
            train_label_temp[i][train_labels[i]] = 1
        for i in range(test_images.shape[0]):
            test_label_temp[i][test_labels[i]] = 1
        train_labels = train_label_temp
        test_labels = test_label_temp

    return train_labels, train_images, test_labels, test_images

def cifar100Data(datasetPath, vecLabel, select = None):
    trainFile = datasetPath + '/train'
    testFile = datasetPath + '/test'

    trainFile = cifarUnpickle(trainFile)
    testFile = cifarUnpickle(testFile)

    trainLabel = trainFile[b'fine_labels']
    trainImage = trainFile[b'data']

    testLabel = testFile[b'fine_labels']
    testImage = testFile[b'data']

    trainImage = torch.from_numpy(trainImage)
    testImage = torch.from_numpy(testImage)
    trainImage = trainImage.reshape(trainImage.shape[0], 3, 32, 32)
    testImage = testImage.reshape(testImage.shape[0], 3, 32, 32)
    trainImage = trainImage.permute(0, 2, 3, 1).numpy()
    testImage = testImage.permute(0, 2, 3, 1).numpy()
    
    if vecLabel:
        trainLabelTemp = torch.zeros(train_labels.shape[0], 100, dtype = torch.float)
        testLabelTemp = torch.zeros(test_labels.shape[0], 100, dtype = torch.float)
        for i in range(train_images.shape[0]):
            trainLabelTemp[i][trainLabel[i]] = 1
        for i in range(test_images.shape[0]):
            testLabelTemp[i][testLabel[i]] = 1
        trainLabel = trainLabelTemp
        testLabel = testLabelTemp

    return trainLabel, trainImage, testLabel, testImage

def prepareData(datasetPath, workPath, vecLabel = False):
    data = cifar10Data(datasetPath, vecLabel)
    data = cifar100Data(datasetPath, vecLabel)
    
    torch.save(data[0],workPath + '/trainLabel')
    torch.save(data[1], workPath + '/trainImage')
    torch.save(data[2], workPath + '/testLabel')
    torch.save(data[3], workPath + '/testImage')
    torch.save(data[2], workPath + '/valLabel')
    torch.save(data[3], workPath + '/valImage')

class CifarSet(torch.utils.data.Dataset):
    def __init__(self, workPath, mode = 'train'):
        super(CifarSet, self).__init__()
        self.mode = mode
        self.image = torch.load(workPath + '/{}Image'.format(mode))
        self.label = torch.load(workPath + '/{}Label'.format(mode))
        self.transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize([224, 224]), transforms.ToTensor()， transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

    def __getitem__(self, index):
        x = self.image[index]
        x = self.transform(x)
        y = self.label[index]
        return x, y
    
    def __len__(self):
        assert self.label.shape[0] == self.image.shape[0]
        return self.label.shape[0]

if __name__ == '__main__':
    pass