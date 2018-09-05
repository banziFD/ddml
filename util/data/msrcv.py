import pickle
import numpy as np 
import glob
import torch
import torch
import torchvision.transforms as transforms

def msrcvData(datasetPath, select = [2, 3, 4, 5, 6, 7, 8]):
    image = np.zeros(len(select) * 30, 224, 224, 3, dtype = np.uint8)
    label = np.zeros(len(select) * 30, dtype = np.uint8)
    idx = 0
    l = 0
    for l in select:
        filename = glob.glob(datasetPath + '/{}*s.bmp'.format(label))
        for f in filename:
            img = Image.open(f)
            img = img.resize(224, 224)
            img = np.array(img.getdata(), dtype = np.uint8)
            img = img.reshape(224, 224, 3)
            image[idx] = img
            label[idx] = l
            idx += 1
    return image, label, image, label

def prepareData(datasetPath, workPath, vecLabel = False):
    data = msrcv1Data(datasetPath, vecLabel)
    
    torch.save(data[0],workPath + '/trainLabel')
    torch.save(data[1], workPath + '/trainImage')
    torch.save(data[2], workPath + '/testLabel')
    torch.save(data[3], workPath + '/testImage')
    torch.save(data[2], workPath + '/valLabel')
    torch.save(data[3], workPath + '/valImage')

class MsrcvSet(torch.utils.data.Dataset):
    def __init__(self, workPath, mode = 'train'):
        super(MarcvSet, self).__init__()
        self.mode = mode
        self.image = torch.load(workPath + '/{}Image'.format(mode))
        self.label = torch.load(workPath + '/{}Label'.format(mode))
        self.transform = transforms.Compose([transforms.ToTensor()， transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])])

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