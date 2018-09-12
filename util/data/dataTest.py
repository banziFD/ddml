from torch.utils.data import DataLoader
from visdom import Visdom
from caltech import CaltechSet
from caltech import prepareData

if __name__ == '__main__':
    workPath = 'd:/git/ddml/ex'
    datasetPath = 'd:/dataset/caltech101/caltech7'
    # prepareData(datasetPath, workPath)
    data = CaltechSet(workPath, mode = 'train', vecLabel = True, nbClass = 7)
    loader = DataLoader(data, shuffle = True)

    vis = Visdom()
    for step, (x, y) in enumerate(loader):
        if(step % 10 == 0):
            print(step, y)
            print(x.shape)
            vis.images(x)
            input('Press Enter\n')
    
