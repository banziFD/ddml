import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from visdom import Visdom
import time
import utils_ddml
import utils_data

def train_noloader(param, work_path, ddml, loss_fn, optimizer, scheduler = None):
    train_images = torch.load(work_path + '/train_images')
    train_labels = torch.load(work_path + '/train_labels')
    val_images = torch.load(work_path + '/val_images')
    val_labels = torch.load(work_path + '/val_labels')
    same_images = torch.load(work_path + '/train_same_images')
    batch = param['batch']
    
    vis = Visdom()
    print('apply training algorithm...')
    
    same_len = same_images.shape[0]
    start = time.time()
    for step in range(param['step'] * param['leap']):
        if param['balance']:
            same = int(batch / 2 * 0.9)
            same = torch.randint(0, same_len, [same,], dtype = torch.long)
            same = same_images[same]
            sameidx1 = same[:, 0].long()
            sameidx2 = same[:, 1].long()
            diff = batch - sameidx1.shape[0]
            idx1 = torch.randint(0, train_images.shape[0], [diff, ], dtype = torch.long)
            idx2 = torch.randint(0, train_images.shape[0], [diff, ], dtype = torch.long)
            idx1 = torch.cat([idx1, sameidx1], 0)
            idx2 = torch.cat([idx2, sameidx2], 0)
            shuffle = torch.randint(0, idx1.shape[0], [batch,], dtype = torch.long)
            idx1 = idx1[shuffle]
            idx2 = idx2[shuffle]
        else:
            idx1 = torch.randint(0, train_images.shape[0], [batch,], dtype = torch.long)
            idx2 = torch.randint(0, train_images.shape[0], [batch,], dtype = torch.long)
        x1 = train_images[idx1]
        x2 = train_images[idx2]
        y1 = train_labels[idx1]
        y2 = train_labels[idx2]
        sim = (y1 == y2).float()
        sim = sim * 2 - 1
        vis.images(x1[0:10], nrow = 1, win = 'win1')
        vis.images(x2[0:10], nrow = 1, win = 'win2')
        print(sim)
        if((step+1) % 2 == 0):
            idx1 = torch.randint(0, val_images.shape[0], [batch,], dtype = torch.long)
            idx2 = torch.randint(0, val_images.shape[0], [batch,], dtype = torch.long)
            x1 = val_images[idx1]
            x2 = val_images[idx2]
            y1 = val_labels[idx1]
            y2 = val_labels[idx2]
            vis.images(x1[0:10], nrow = 1, win = 'win3')
            vis.images(x2[0:10], nrow = 1, win = 'win4')
            sim = (y1 == y2).float()
            
def train(param, work_path, ddml, train_loader, val_loader, loss_fn, optimizer, scheduler = None):
    print('apply training algorithm...')
    log = open(work_path + '/log.txt', 'w')
    log.write('epoch time training_loss \n')
    vis = Visdom()
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
                handle1 = ddml.feature.linear1.register_forward_hook(make_hook('f', inter_feature))
                handle2 = ddml.feature.linear2.register_forward_hook(make_hook('f', inter_feature))
                handle3 = ddml.feature.linear3.register_forward_hook(make_hook('f', inter_feature))
            x1.requires_grad = False;
            x2.requires_grad = False;
            if(param['gpu']):
                x1 = x1.cuda()
                x2 = x2.cuda()
                sim = sim.cuda()
            optimizer.zero_grad()
            y = ddml(x1, x2)
            loss = loss_fn(y, sim)
            l2_reg = torch.tensor(0.0).cuda()
            for p in ddml.parameters():
                l2_reg += torch.norm(p)
            loss += param['lambda']*l2_reg
            
            error_train = error_train + loss.data.item()
            loss.backward()
            optimizer.step()

            if step == 0:
                inter_feature = [item[0][0:10] for item in inter_feature]
                temp = inter_feature[5]
                inter_feature = inter_feature[0:3]
                inter_feature[2] = inter_feature[2] - temp
                inter_feature[0] = inter_feature[0].reshape(-1, 3, 32, 32)
                inter_feature[1] = inter_feature[1].reshape(-1, 1, 16, 16)
                inter_feature[2] = inter_feature[2].reshape(-1, 1, 16, 16)
                vis.images(inter_feature[0], nrow = 1, opts = {'caption' : win1}, win = win1)
                vis.images(inter_feature[1] * 100, nrow = 1, opts = {'caption' : win2}, win = win2)
                vis.images(inter_feature[2] * 100, nrow = 1, opts = {'caption' : win3}, win = win3)
                del inter_feature
                handle1.remove()
                handle2.remove()
                handle3.remove()
        error_train = error_train / (step + 1)

        for step, (x1, x2, sim) in enumerate(val_loader):
            x1.requires_grad = False;
            x2.requires_grad = False;
            if(param['gpu']):
                x1 = x1.cuda()
                x2 = x2.cuda()
                sim = sim.cuda()
            y = ddml(x1, x2)
            loss = loss_fn(y, sim)
            l2_reg = 0
            for p in ddml.parameters():
                l2_reg += torch.norm(p)
            loss += param['lambda'] * l2_reg
            error_val = error_val + loss.data.item()
        error_val = error_val / (step + 1)

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
    for step, (x1, x2, sim, y1, y2) in enumerate(loader):


def main_loader():
    path = set_path()
    param = set_param()
    utils_data.prepare_data(path['dataset_path'], path['work_path'], False)

    train_set = utils_data.DDMLData(path['work_path'], 0, 4096)
    train_loader = DataLoader(train_set, batch_size = param['batch'], shuffle = True)
    val_set = utils_data.DDMLData(path['work_path'], 2, 512)
    val_loader = DataLoader(val_set, batch_size = param['batch'], shuffle = True)

    ddml = utils_ddml.DDML()
    loss_fn = utils_ddml.DDMLLoss(param['tau'], param['beta'])
    optimizer = torch.optim.Adam(ddml.parameters(), lr = param['lr'])
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, param['milestones'], gamma = 0.5)

    train(param, path['work_path'], ddml, train_loader, val_loader, loss_fn, optimizer)
    torch.save(ddml, path['work_path'] + '/ddml')

    ddml = torch.load(path['work_path'] + '/ddml')
    test_set = utils_data.DDMLData(path['work_path'], 1, 4096, False)
    test_loader = DataLoader(test_set)
    result = test(param, path['work_path'], ddml, test_loader, param['tau'])
    torch.save(result, path['work_path'] + '/result')

def main_noloader():
    path = set_path()
    param = set_param()
    # utils_data.prepare_data(path['dataset_path'], path['work_path'], param['balance'])
    
    ddml = utils_ddml.DDML()
    # loss_fn = utils_ddml.DDMLLoss(param['tau'], param['beta'])
    # optimizer = torch.optim.Adam(ddml.parameters(), lr = param['lr'])
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, param['milestones'], gamma = 0.5)

    # train_noloader(param, path['work_path'], ddml, loss_fn, optimizer)
    # torch.save(ddml, path['work_path'] + '/ddml')

    # ddml = torch.load(path['work_path'] + '/ddml')
    # test_set = utils_data.DDMLData(path['work_path'], 1, 4096)
    # test_loader = DataLoader(test_set)
    # result = test(param, path['work_path'], ddml, test_loader, param['tau'])
    # torch.save(result, path['work_path'] + '/result')
    
    
    
def set_param():
    param = dict()
    param['lr'] = 0.00005
    param['batch'] = 128
#     param['epoch'] = 100
    param['step'] = 1
    param['leap'] = 1
    param['gpu'] = False
    param['tau'] = 3
    param['beta'] = 1
    param['lambda'] = 0.2
    param['balance'] = True
    param['milestones'] = [30, 60, 90, 120, 150, 180, 210]
    param['show'] = 256
    return param

def set_path():
    dataset_path = 'd:\dataset\cifar-10-python'
    work_path = 'd:\git\ddml\ex'
    return {'dataset_path': dataset_path, 'work_path': work_path}

if __name__ == '__main__':
    main_noloader()