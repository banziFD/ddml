from torch.utils.data import DataLoader
from visdom import Visdom
from caltech import CaltechSet
from cifar import CifarSet
from cifar import prepareData100
from msrcv import prepareData
from msrcv import MsrcvSet

if __name__ == '__main__':
    workPath = 'd:/git/ddml/ex'
    datasetPath = 'd:/dataset/msrcv7'
    # prepareData100(datasetPath, workPath)
    # data = CaltechSet(workPath, mode = 'train', vecLabel = True, nbClass = 7)
    # data = CifarSet(workPath, mode = 'train', vecLabel = False, nbClass = 100)
    prepareData(datasetPath, workPath)
    data = MsrcvSet(workPath, mode = 'train', vecLabel = False, nbClass = 7)
    loader = DataLoader(data, shuffle = True)

    vis = Visdom()
    for step, (x, y) in enumerate(loader):
        if(step % 10 == 0):
            print(step, y)
            print(x.shape)
            vis.images(x)
            input('Press Enter\n')
    