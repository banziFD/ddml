import os
from PIL import Image
import torch
import torchvision.transforms as transforms

class DaiSet(torch.utils.data.Dataset):
    def __init__(self, image_path, data, mode = 'box'):
        super(DaiSet, self).__init__()
        self.data = data
        self.image_path = image_path
        self.mode = mode
        self.transform = transforms.Compose(
            [transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])]
        )

    def __getitem__(self, index):
        key = str(index)
        ori_img = self.data[key]['ori_img']
        ori_img = ori_img + '.jpg'
        ori_img = os.path.join(self.image_path, ori_img)
        ori_img = Image.open(ori_img)
        sub_img = ori_img.crop(self.data[key][self.mode])
        image = self.transform(sub_img)
        return index, image

    def __len__(self):
        return len(self.data.keys())