import random

import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image

import os


class AlphaDataset(Dataset):
    def __init__(self, csvfile, root_dir):
        self.data_table = pd.read_csv(csvfile, header=None)
        self.root_dir = os.path.abspath(root_dir)

    def __len__(self):
        return len(self.data_table)

    def transform(self, image, mask):
        bands = image.split()
        bands = [b.resize(size=(256, 256), resample=Image.NEAREST) for b in bands]
        image = Image.merge('RGBA', bands)
        mask = mask.resize(size=(256, 256), resample=Image.NEAREST)

        if random.random() > 0.5:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        if random.random() > 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        image = TF.to_tensor(image)
        mask = TF.to_tensor(mask)

        return image, mask[0, :, :].expand(4, -1, -1)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.data_table.iloc[idx, 0], self.data_table.iloc[idx, 1])
        mask_path = os.path.join(self.root_dir, self.data_table.iloc[idx, 0], self.data_table.iloc[idx, 2])
        image = Image.open(img_path)
        mask = Image.open(mask_path)
        image, mask = self.transform(image, mask)
        return image * mask, mask, image


class AlphaDatasetMaskOnly(Dataset):
    def __init__(self, csvfile, root_dir, size=(256, 256), channels=4):
        self.data_table = pd.read_csv(csvfile, header=None)
        self.root_dir = os.path.abspath(root_dir)
        self.size = size
        self.channels = channels

    def __len__(self):
        return len(self.data_table)

    def transform(self, mask):
        mask = mask.resize(size=self.size, resample=Image.NEAREST)

        if random.random() > 0.5:
            mask = TF.vflip(mask)

        if random.random() > 0.5:
            mask = TF.hflip(mask)

        mask = TF.to_tensor(mask)

        return mask[0, :, :].expand(self.channels, -1, -1)

    def __getitem__(self, idx):
        mask_path = os.path.join(self.root_dir, self.data_table.iloc[idx, 0], self.data_table.iloc[idx, 2])
        mask = Image.open(mask_path)
        mask = self.transform(mask)
        return mask
