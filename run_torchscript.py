import torch
import torchvision
import torchvision.transforms as transforms
import os
import argparse

import PIL.Image as Image

import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('script', help='Path to the torchscript file')
    parser.add_argument('image', help='Path to the image to inpaint')
    parser.add_argument('mask', help='Path to the mask file')
    args = parser.parse_args()

    if not os.path.isfile(args.script):
        print(f'{args.script}: file does not exist')
        return
    if not os.path.isfile(args.image):
        print(f'{args.image}: file does not exist')
        return
    if not os.path.isfile(args.mask):
        print(f'{args.mask}: file does not exist')
        return

    model = torch.jit.load(args.script)
    model.eval()

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    img = Image.open(args.image).convert('RGB')
    mask = Image.open(args.mask).convert('RGB')

    tensor_img = transform(img).unsqueeze(0)
    tensor_mask = transform(mask).unsqueeze(0)

    with torch.no_grad():
        prediction, _ = model(tensor_img, tensor_mask)

    plt.figure(figsize=(8, 4))
    plt.axis('off')
    plt.title(f'{args.script} - {args.image}')
    images = [tensor_mask[0, :, :, :], tensor_img[0, :, :, :], prediction[0, :, :, :]]
    plt.imshow(torchvision.utils.make_grid(images, nrow=3, padding=2, pad_value=1, normalize=True).permute(1, 2, 0))
    plt.show()

if __name__ == '__main__':
    main()
