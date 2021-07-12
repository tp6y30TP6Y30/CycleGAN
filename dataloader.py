import os
from os import listdir
from os.path import join
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
from tqdm import tqdm
import random
import torch

class CycleGANData(Dataset):
    def __init__(self, query, mode, target):
        super(CycleGANData, self).__init__()
        self.target = target
        self.img_path = join('./datasets/', query, mode + target)
        self.img_list = listdir(self.img_path)
        self.transform = transforms.Compose([
                            transforms.Resize((256, 256)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                         ])

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        img = Image.open(join(self.img_path, self.img_list[index])).convert('RGB')
        img = self.transform(img)
        label = 0 if self.target == 'A' else 1
        return img, label

if __name__ == '__main__':
    dataset = CycleGANData('vangogh2photo', 'train', 'A')
    loader = DataLoader(dataset, batch_size = 2, shuffle = True)
    for img, label in tqdm(loader):
        print(img.shape)
        print(label)
        break