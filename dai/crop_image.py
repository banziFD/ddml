from PIL import Image
import os
import json

def crop_box(data, image_path, work_path):
    for key in data.keys():
        if key % 100 == 0:
            print("ground_truth: {}".format(key))
        ori_img = data[key]['ori_img']
        ori_img = ori_img + '.jpg'
        ori_img = os.path.join(image_path, ori_img)
        ori_img = Image.open(ori_img)
        sub_img = ori_img.crop(data[key]['box'])
        sub_img.save(os.path.join(work_path, str(key) + '.jpg'), 'JPEG')


def crop_proposal(data, image_path, work_path):
    for key in data.keys():
        if key % 100 == 0:
            print("proposal: {}".format(key))
        ori_img = data[key]['ori_img']
        ori_img = ori_img + '.jpg'
        ori_img = os.path.join(image_path, ori_img)
        ori_img = Image.open(ori_img)
        sub_img = ori_img.crop(data[key]['proposal'])
        sub_img.save(os.path.join(work_path, str(key) + '.jpg'), 'JPEG')

if __name__ == "__main__":
    # ground_truth = json.load(open('ground_truth.json', 'r'))
    proposal = json.load(open('proposal.json', 'r'))
    crop_proposal(proposal, '.', 'work_path')

