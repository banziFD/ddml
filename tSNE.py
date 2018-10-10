import numpy as np  
from sklearn.manifold import TSNE
from visdom import Visdom   

if __name__ == '__main__':
    feature = np.load('trainfeature.npy')
    label = np.load('trainlabel.npy')

    feaEmb = TSNE(n_components = 2).fit_transform(feature)
    print('Complete embedded')

    vis = Visdom()
    
    vis.scatter(feaEmb, label)
    