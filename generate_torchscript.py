import torch
import pconv.modules as pcmodules
import os
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Path to the checkpoint file used to initialize the model weights')
    parser.add_argument('script', help='Name of the torchscript file to generate')

    args = parser.parse_args()
    if not os.path.isfile(args.checkpoint):
        print(f'{args.checkpoint}: file does not exist')
        return

    checkpoint = torch.load(args.checkpoint)

    unet = pcmodules.UNet()
    unet.load_state_dict(checkpoint['model_state_dict'])
    unet.eval()

    scripted_module = torch.jit.script(unet)
    scripted_module.save(args.script)
    print(f'Saved torchscript file to {args.script}')


if __name__ == '__main__':
    main()
