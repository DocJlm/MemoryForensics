# =====================================================
# network/data.py - 数据加载
# =====================================================
from torch.utils.data import Dataset
from PIL import Image

class SingleInputDataset(Dataset):
    def __init__(self, txt_path, train_transform=None, valid_transform=None):
        lines = open(txt_path, 'r')
        imgs = []
        for line in lines:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.train_transform = train_transform
        self.valid_transform = valid_transform

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.train_transform is not None:
            img = self.train_transform(img)
        if self.valid_transform is not None:
            img = self.valid_transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)


class TestDataset(Dataset):
    def __init__(self, txt_path, test_transform=None):
        lines = open(txt_path, 'r')
        imgs = []
        for line in lines:
            line = line.rstrip()
            words = line.split()
            imgs.append((words[0], int(words[1])))

        self.imgs = imgs
        self.test_transform = test_transform

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')
        
        if self.test_transform is not None:
            img = self.test_transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)
