import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os, sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt

BS = opt.bs
print("Batch size:", BS)
crop_size = 'whole_img'
if opt.crop:
    crop_size = opt.crop_size


def tensorShow(tensors, titles=None):
    fig = plt.figure()
    for tensor, tit, i in zip(tensors, titles, range(len(tensors))):
        img = make_grid(tensor)
        npimg = img.numpy()
        ax = fig.add_subplot(211 + i)
        ax.imshow(np.transpose(npimg, (1, 2, 0)))
        ax.set_title(tit)
    plt.show()


class RESIDE_Dataset(data.Dataset):
    def __init__(self, path, train, size=crop_size):
        super(RESIDE_Dataset, self).__init__()
        self.size = size
        self.train = train
        print(f'📁 Dataset path: {path}')
        print('Crop size:', size)

        self.haze_imgs_dir = os.listdir(os.path.join(path, 'hazy'))
        self.haze_imgs = [os.path.join(path, 'hazy', img) for img in self.haze_imgs_dir]
        self.clear_dir = os.path.join(path, 'clear')

        self.clear_map = {}
        for fname in os.listdir(self.clear_dir):
            cid = fname.split('_')[0].split('.')[0]
            self.clear_map[cid] = os.path.join(self.clear_dir, fname)

    def __getitem__(self, index):
        haze_path = self.haze_imgs[index]
        haze = Image.open(haze_path)

        if isinstance(self.size, int):
            while haze.size[0] < self.size or haze.size[1] < self.size:
                index = random.randint(0, len(self.haze_imgs) - 1)
                haze_path = self.haze_imgs[index]
                haze = Image.open(haze_path)

        basename = os.path.basename(haze_path)
        id = basename.split('_')[0].split('.')[0]

        if id not in self.clear_map:
            raise FileNotFoundError(f"Clear image for {basename} not found in {self.clear_dir}")

        clear = Image.open(self.clear_map[id])
        clear = tfs.CenterCrop(haze.size[::-1])(clear)

        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)

        haze, clear = self.augData(haze.convert("RGB"), clear.convert("RGB"))
        return haze, clear

    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        data = tfs.ToTensor()(data)
        data = tfs.Normalize(mean=[0.64, 0.6, 0.58], std=[0.14, 0.15, 0.152])(data)
        target = tfs.ToTensor()(target)
        return data, target

    def __len__(self):
        return len(self.haze_imgs)


def get_dataset_paths(trainset_name):
    name = trainset_name.lower()
    if 'its' in name:
        subdir = 'ITS'
    elif 'ots' in name:
        subdir = 'OTS'
    elif 'sots' in name:
        subdir = 'SOTS'
    elif 'nh4k' in name or 'nh-haze4k' in name:
        subdir = 'NH-HAZE4K'
    elif 'nh-haze' in name:
        subdir = 'NH-HAZE'
    elif 'haze4k' in name:
        subdir = 'HAZE4K'
    else:
        raise ValueError(f"❌ Unknown dataset name: {trainset_name}")

    base_path = os.path.join(os.path.dirname(os.getcwd()), 'data', 'RESIDE', subdir)
    train_path = os.path.join(base_path, 'train')
    test_path = os.path.join(base_path, 'test')
    return train_path, test_path


train_path, test_path = get_dataset_paths(opt.trainset)
print(f"✅ Using dataset: {train_path}")

loader_train = DataLoader(
    dataset=RESIDE_Dataset(train_path, train=True, size=crop_size),
    batch_size=BS,
    shuffle=True,
    pin_memory=True
)

loader_test = DataLoader(
    dataset=RESIDE_Dataset(test_path, train=False, size='whole_img'),
    batch_size=1,
    shuffle=False,
    pin_memory=True
)

if __name__ == "__main__":
    print(f"Train path: {train_path}")
    print(f"Test path:  {test_path}")
    print("Data loaders ready ✅")