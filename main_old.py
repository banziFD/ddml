import numpy as np
import torch
from torch.utils.data import DataLoader
from visdom import Visdom
import time
import utils_ddml
import utils_data
from cifar.utils_model import CNN as Feature

def set_param():
    param = dict()
    param['lr'] = 0.001
    param['batch'] = 512
    param['epoch'] = 100
    param['gpu'] = True
    param['tau'] = 5
    param['beta'] = 5
    param['sample_size'] = 10000
    param['milestones'] = [50, 100, 150, 200, 500, 600, 700]
    return param

def set_path():
#     dataset_path = 'd:\dataset\cifar-10-python'
#     work_path = 'd:\git\ddml\ex'
    dataset_path = '/home/spyisflying/dataset/cifar/cifar-10-python'
    work_path = '/home/spyisflying/git/ddml/ex'
    return {'dataset_path': dataset_path, 'work_path': work_path}

def make_hook(flag, data):
    if(flag == 'f'):
        def hook(m, input, output):
            data.append(input)
        return hook
    if(flag == 'b'):
        def hook(m, input, output):
            data.append(output)
        return hook

  


def train(param, work_path, ddml, train_loader, val_loader, loss_fn, optimizer, scheduler = None):
    print('apply training algorithm...')
    log = open(work_path + '/log.txt', 'w')
    log.write('epoch time training_loss \n')
    vis = Visdom()
    win0 = "origin"
    win1 = "layer1"
    win2 = "layer2"
    win3 = "layer3"
    win4 = "curve"
    curveX = torch.zeros(param['epoch'] * 2, 2)
    curveY = torch.ones(param['epoch'] * 2, dtype = torch.long)
    if(param['gpu']):
        ddml = ddml.cuda()
        loss_fn = loss_fn.cuda()
    for epoch in range(param['epoch']):
        if not scheduler is None:
            scheduler.step() 
        start = time.time()
        error_train = 0
        error_val = 0
        for step, (x1, x2, sim) in enumerate(train_loader):
            if(step == 0):
                inter_feature = []
                handle1 = ddml.linear1.register_forward_hook(make_hook('f', inter_feature))
                handle2 = ddml.linear2.register_forward_hook(make_hook('f', inter_feature))
                handle3 = ddml.linear3.register_forward_hook(make_hook('f', inter_feature))
            x1.requires_grad = False;
            x2.requires_grad = False;
            if(param['gpu']):
                x1 = x1.cuda()
                x2 = x2.cuda()
                sim = sim.cuda()
            optimizer.zero_grad()
            y = ddml(x1, x2)
            loss = loss_fn(y, sim)
            error_train = error_train + loss.data.item()
            loss.backward()
            optimizer.step()
            if step == 0:
                inter_feature = [item[0][0:10] for item in inter_feature]
                inter_feature = inter_feature[0:3]
                inter_feature[0] = inter_feature[0].reshape(-1, 3, 32, 32)
                inter_feature[1] = inter_feature[1].reshape(-1, 1, 32, 32)
                inter_feature[2] = inter_feature[2].reshape(-1, 1, 32, 32)
                vis.images(x1[0:10], nrow = 1, opts = {'caption': win0}, win = win0)
                vis.images(inter_feature[0], nrow = 1, opts = {'caption' : win1}, win = win1)
                vis.images(inter_feature[1] * 100, nrow = 1, opts = {'caption' : win2}, win = win2)
                vis.images(inter_feature[2] * 100, nrow = 1, opts = {'caption' : win3}, win = win3)
                del inter_feature
                handle1.remove()
                handle2.remove()
                handle3.remove()
        
        for step, (x1, x2, sim) in enumerate(val_loader):
            x1.requires_grad = False;
            x2.requires_grad = False;
            if(param['gpu']):
                x1 = x1.cuda()
                x2 = x2.cuda()
                sim = sim.cuda()
            y = ddml(x1, x2)
            loss = loss_fn(y, sim)
            error_val = error_val + loss.data.item()
        error_val = error_val * 10
        curveX[epoch * 2][0] = epoch
        curveX[epoch * 2][1] = error_train
        curveX[epoch * 2 + 1][0] = epoch
        curveX[epoch * 2 + 1][1] = error_val
        curveY[epoch * 2] = 1
        curveY[epoch * 2 + 1] = 2
        
        if(epoch > 1):
            x = curveX[0 : epoch * 2 + 1]
            y = curveY[0 : epoch * 2 + 1]
            vis.scatter(x, y, win = win4)
        current_line = [epoch, time.time() - start, error_train,  error_val]
        print(current_line)
        log.write(str(current_line) + '\n')
        
    log.close()


def test(param, work_path, ddml, loader, tau):
    print('testing...')
    if(param['gpu']):
        ddml = ddml.cuda()
    result = torch.zeros(int(loader.dataset.__len__()), 5)
    idx = 0
    count = 0
    for step, (x1, x2, sim, y1, y2) in enumerate(loader):
        if param['gpu']:
           x1 = x1.cuda()
           x2 = x2.cuda()
           sim = sim.cuda()
        y = ddml(x1, x2)
        distance = torch.norm(y, 2, 1)
        if(distance >= tau):
            pred = -1
        else:
            pred = 1
        result[idx, 0] = pred
        result[idx, 1] = sim
        result[idx, 2] = y1
        result[idx, 3] = y2
        result[idx, 4] = distance
        if pred == sim:
            count += 1
        idx += 1
    print('accuracy: ', count / idx)
    return result


if __name__ == '__main__':
    path = set_path()
    param = set_param()
    utils_data.prepare_data(path['dataset_path'], path['work_path'])
    train_set = utils_data.DDMLData(path['work_path'], 0)
    train_loader = DataLoader(train_set, batch_size = param['batch'], shuffle = True)
    val_set = utils_data.DDMLData(path['work_path'], 2)
    val_loader = DataLoader(val_set, batch_size = param['batch'], shuffle = True)
    ddml = utils_ddml.DDML()
    loss_fn = utils_ddml.DDMLLoss(param['tau'], param['beta'])
    optimizer = torch.optim.Adam(ddml.parameters(), lr = param['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, param['milestones'], gamma = 0.1)
    train(param, path['work_path'], ddml, train_loader, val_loader, loss_fn, optimizer)
    torch.save(ddml, path['work_path'] + '/ddml')
   
    