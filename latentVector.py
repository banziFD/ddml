import numpy as np  
import torch
from torch.utils.data import DataLoader
import utils_ddml
import utils_data

def latentVector(pa, workPath, ddml, loader):
    size = loader.dataset().shape[0]
    feature = torch.zeros(size, 512)
    for step, (x, label) in enumerate(loader):

def setParam():
    pass

def setPath():
    pass

if __name__ == '__main__':
    pa = setParam()
    path = setPath()
    
    print('Preparing data...')
    # utils_data.prepareData(path['datasetPath'], path['workPath'])

    ddml = torch.load(path['workPath'] + '/ddml')

    latentVector = latentVector()