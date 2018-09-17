from torch.utils.data import DataLoader
from visdom import Visdom
from caltech import CaltechSet
from caltech import prepareData
# from cifar import CifarSet
# from cifar import prepareData100
# from msrcv import prepareData
# from msrcv import MsrcvSet

if __name__ == '__main__':
    workPath = 'd:/git/ddml/ex'
    datasetPath = 'd:/dataset/caltech7'
    # prepareData100(datasetPath, workPath)
    # prepareData(datasetPath, workPath)
    data = CaltechSet(workPath, mode = 'train', vecLabel = False, nbClass = 7)
    # data = CifarSet(workPath, mode = 'train', vecLabel = False, nbClass = 100)
    
    # data = MsrcvSet(workPath, mode = 'train', vecLabel = False, nbClass = 7)
    loader = DataLoader(data, shuffle = True, num_workers = 2)

    stat = dict()
    for i in range(7):
        stat[i] = 0
    vis = Visdom()
    for step, (x, y) in enumerate(loader):
        stat[y.data.item()] += 1
        if(step % 100 == 0):
            print(step, y)
            print(x.shape)
            vis.images(x, win = '1')
            input('Press Enter\n')
    print(stat)