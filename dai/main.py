import json
import time
from glob import glob
import os
import torch
from torch.utils.data import DataLoader
import numpy as np

import crop_image
import res_feature
import voc_data


def set_param():
    param = dict()
    param['gpu'] = True
    param['batch_size'] = 32
    return param

def prepare(image_path, mat_path, work_path):
    mat_list = glob(mat_path + "/*.mat")
    mat_list.sort()
    gt = parse_mat.ground_truth(mat_list)
    json.dump(gt, open(os.path.join(work_path, 'ground_truth.json'), 'w'))
    ps = parse_mat.proposal(mat_list)
    json.dump(ps, open(os.path.join(work_path, 'proposal.json'), 'w'))

def extract_feature(param, work_path, loader):
    res = res_feature.ResFeature()
    for pa in res.parameters():
        pa.requires_grad = False

    if param['gpu']:
        res = res.cuda()

    res.eval()

    for step, (key, image) in enumerate(loader):
        start = time.time()
        if param['gpu']:
            image = image.cuda()
        feature = res(image)
        feature = feature.detach().cpu()
        feature = feature.numpy()
        for idx in range(key.shape[0]):
            np.save(os.path.join(work_path, str(key[idx])), feature[idx])
        print('Operate on {}    time {}'.format(step, time.time() - start))

def main_g():
    image_path = './data/JPEGImages'
    mat_path = './data/vocTrainFeats'
    work_path = './work_path/ground_truth'

    param = set_param()

    metadata = json.load(open(work_path + '/ground_truth.json', 'r'))

    dataset = voc_data.DaiSet(image_path, metadata, mode = 'box')
    print('dataset size: {}'.format(dataset.__len__()))
    loader = DataLoader(dataset, batch_size = param['batch_size'], num_workers=4)
    extract_feature(param, work_path, loader)
    print('compelete!')

def main_p():
    image_path = './data/JPEGImages'
    mat_path = './data/vocTrainFeats'
    work_path = './work_path/proposal'

    param = set_param()

    metadata = json.load(open(work_path + '/proposal.json', 'r'))

    dataset = voc_data.DaiSet(image_path, metadata, mode = 'proposal')
    print('dataset size: {}'.format(dataset.__len__()))
    loader = DataLoader(dataset, batch_size = param['batch_size'], num_workers=4)
    extract_feature(param, work_path, loader)
    print('compelete!')

if __name__ == '__main__':
    main_p()
    main_g()