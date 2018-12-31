import json
from glob import glob
from scipy.io import loadmat

def ground_truth(mat_list):
    result = dict()
    primary_key = 0

    for mat in mat_list:
        ori_img = mat[-10:-4]
        mat = loadmat(mat)
        obj_list = mat['obj_name']
        box_list = mat['gtbox']

        for idx in range(box_list.shape[0]):
            current = dict()
            current['ori_img'] = ori_img
            current['obj'] = obj_list[0][idx][0]
            current['box'] = box_list[idx].tolist()
            result[primary_key] = current
            primary_key += 1

    return result

def proposal(mat_list):
    result = dict()
    primary_key = 0

    for mat in mat_list:
        ori_img = mat[-10:-4]
        mat = loadmat(mat)
        proposal = mat['boxes']
        for idx in range(proposal.shape[0]):
            current = dict()
            current['ori_img'] = ori_img
            current['proposal'] = proposal[idx].tolist()
            result[primary_key] = current
            primary_key += 1
    return result

if __name__ == "__main__":
    # mat_list = glob('/Users/banzifd/dai/VOCTrainFeats/*.mat')
    mat_list = ['./000005.mat']
    gt = ground_truth(mat_list)
    ps = proposal(mat_list)
    json.dump(gt, open('ground_truth.json', 'w'))
    json.dump(ps, open('proposal.json', 'w'))