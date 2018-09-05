import pickle
import numpy as np 
import glob
import torch
import torch
import torchvision.transforms as transforms

def caltechData(datasetPath, select = [2, 3, 4, 5, 6, 7, 8]):
    
def prepareData(datasetPath, workPath, vecLabel = False):
    data = msrcv1Data(datasetPath, vecLabel)
    
    torch.save(data[0],workPath + '/trainLabel')
    torch.save(data[1], workPath + '/trainImage')
    torch.save(data[2], workPath + '/testLabel')
    torch.save(data[3], workPath + '/testImage')
    torch.save(data[2], workPath + '/valLabel')
    torch.save(data[3], workPath + '/valImage')

class CaltechSet(torch.utils.data.Dataset):
    pass

if __name__ == '__main__':
    pass