import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.dataloader import DataLoader

import os
import argparse

import PIL.Image as Image

import utils.mask

import matplotlib.pyplot as plt
import numpy as np
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--torchscript', help='Path to the torchscript files used to generate the samples',
                        nargs='+', required=True)
    parser.add_argument('-d', '--dataset', help='Path to the image dataset to use', required=True)
    parser.add_argument('-m', '--mask', help='Path to the mask dataset to use', required=True)
    parser.add_argument('-s', '--seed', help='Seed to the random number generator', type=int)
    parser.add_argument('-n', '--nsamples', help='Number of samples to generate', default=100, type=int)
    parser.add_argument('-o', '--output', help='Path where to store the sample files', default='.', type=str)
    args = parser.parse_args()

    for ts in args.torchscript:
        if not os.path.isfile(ts):
            print(f'{args.script}: file does not exist')
            return
    if not os.path.isdir(args.dataset):
        print(f'{args.image}: not a directory')
        return
    if not os.path.isdir(args.mask):
        print(f'{args.mask}: not a directory')
        return
    if not os.path.isdir(args.output):
        print(f'{args.output}: not a directory')
        return

    if args.seed:
        torch.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)

    # Load models
    unets = []
    for ts in args.torchscript:
        unets.append(torch.jit.load(ts))
        unets[-1].eval()

    # Prepare dataloaders
    image_size = 256
    
    data_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomCrop(image_size, pad_if_needed=True, padding_mode='reflect'),
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomAffine(degrees=90, translate=(0.2, 0.2), shear=(-15, 15, -15, 15)),
        transforms.Resize(size=(image_size, image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: (x == 0).float())
    ])

    sample_mask_loader = utils.mask.MaskLoader(args.mask, transform=mask_transform)
    sample_dataset = utils.mask.ImageFolderWithMaskLoader(args.dataset, sample_mask_loader, transform=data_transform)

    print(f'Sample data size is {len(sample_dataset)}')

    sample_dataloader = DataLoader(sample_dataset, batch_size=1, shuffle=True, num_workers=1)

    # Iterate and store sample images

    sample_data_iterator = iter(sample_dataloader)
    with torch.no_grad():
        for i in range(args.nsamples):
            masked_image, mask, gt = next(sample_data_iterator)
            m = mask[0, :, :, :]
            g = gt[0, :, :, :]
            imgs = [masked_image[0, :, :, :] + (1 - m), g]

            for unet in unets:
                prediction, _ = unet(masked_image, mask)
                imgs.append(m * g + (1 - m) * prediction[0, :, :, :])

            grid = torchvision.utils.make_grid(imgs, nrow=len(imgs), padding=2, pad_value=1, range=(0, 1))
            torchvision.utils.save_image(grid, os.path.join(args.output, 'sample_{:03d}.png'.format(i)))


if __name__ == '__main__':
    main()
