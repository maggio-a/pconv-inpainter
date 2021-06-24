import torch
import torch.optim as optim
from torch.utils.data import Subset
from torch.utils.data.dataloader import DataLoader

import torchvision.utils
import torchvision.transforms as transforms

import pconv.modules as pcmodules
import pconv.layers as pclayers

import utils.mask
from utils.metering import Timer, RunningAverage

import random
#import matplotlib.pyplot as plt
import numpy as np

import os
import os.path
import shutil


class Args:
    def __init__(self):
        self.seed = None

        self.gpu = True

        self.train_dataset_path = 'path_to_train_dataset'
        self.test_dataset_path = 'path_to_test_dataset'
        self.mask_dataset_path = 'path_to_mask_dataset'

        self.nepochs = 1000
        self.samples_per_epoch = 30000
        self.samples_per_validation = 5000

        self.batch_size = 6

        self.load_checkpoint = None
        self.save_checkpoint_dir = './checkpoints'
        self.coarse_checkpoint_step = 5

        self.mode = 'train'  # either 'train' or 'test'
        self.tuning_phase = False

        self.useAMP = False


def main():
    args = Args()

    if args.seed:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    assert(args.mode == 'train' or args.mode == 'test')

    if os.path.exists(args.save_checkpoint_dir):
        assert os.path.isdir(args.save_checkpoint_dir)
    else:
        os.makedirs(args.save_checkpoint_dir)

    device = torch.device('cuda:0' if torch.cuda.is_available() and args.gpu else 'cpu')

    print(f'Device is {device}')
    if torch.cuda.is_available():
        print(f'Running cuda on GPU: {torch.cuda.get_device_name(0)}')

    ################################################
    # Load and initialize model, optimizer and state
    ################################################

    model = pcmodules.UNet()
    model.to(device)

    loss_function = pcmodules.IrregularHolesLoss()
    loss_function.to(device)

    lr = 0.0002

    history = []

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=args.useAMP)

    current_epoch = 0

    if args.load_checkpoint:
        if os.path.isfile(args.load_checkpoint):
            print(f'Loading checkpoint file {args.load_checkpoint}')
            checkpoint = torch.load(args.load_checkpoint, map_location=device)

            current_epoch = checkpoint['current_epoch']
            model.load_state_dict(checkpoint['model_state_dict'])
            if len(checkpoint['scaler_state_dict']) > 0:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            history = checkpoint['history']
        else:
            print(f'Checkpoint file {args.load_checkpoint} not found')

    if args.tuning_phase:
        # adjust learning rate
        lr = 0.00005

        # freeze encoding batch normalization
        for block in model.modules():
            if type(block) == pcmodules.UNet.EncoderBlock:
                for layer in block.children():
                    print('Freezing layer ', layer)
                    if type(layer) == torch.nn.BatchNorm2d:
                        for parameter in layer.parameters():
                            parameter.requires_grad = False

        # when finetuning clear the optimization state
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
        scaler = torch.cuda.amp.GradScaler(enabled=args.useAMP)

    ##############################
    # Build dataset and DataLoader
    ##############################

    image_size = 512

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

    train_mask_loader = utils.mask.MaskLoader(args.mask_dataset_path, transform=mask_transform)
    test_mask_loader = utils.mask.MaskLoader(args.mask_dataset_path, transform=mask_transform)

    train_dataset = utils.mask.ImageFolderWithMaskLoader(args.train_dataset_path, train_mask_loader, transform=data_transform)
    test_dataset = utils.mask.ImageFolderWithMaskLoader(args.test_dataset_path, test_mask_loader, transform=data_transform)

    train_size = len(train_dataset)
    test_size = len(test_dataset)

    train_dl = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)
    test_dl = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=1)

    print(f'Training data size is {train_size}')
    print(f'Test data size is {test_size}')

    ####################################
    # Store and visualize some test data
    ####################################

    #model.eval()

    #imglist = []
    #a = []
    #b = []
    #c = []

    #test_data_iterator = iter(test_dl)

    #for i in range(6):
    #    masked_img, mask, img = map(lambda x: x[0, :, :, :], next(test_data_iterator))
    #    with torch.no_grad():
    #        pred, _ = model(masked_img.unsqueeze(0).to(device), mask.unsqueeze(0).to(device))
    #    imglist.append(masked_img + (1 - mask))
    #    imglist.append(img)
    #    imglist.append(pred[0, :, :, :].detach().cpu())
    #    a.append(masked_img)
    #    b.append(mask)
    #    c.append(img)

    #model.train()

    #plt.figure(figsize=(16, 8))
    #plt.axis('off')
    #plt.title('MaskedImageFolder')

    #plt.imshow(np.transpose(torchvision.utils.make_grid(imglist,
    #                                                    padding=2, normalize=False, nrow=3), (1, 2, 0)))
    #plt.show()

    #test_tensors = torch.stack(a).to(device), torch.stack(b).to(device), torch.stack(c).to(device)

    ###################
    # execute the model
    ###################

    if args.mode == 'train':
        for epoch in range(args.nepochs):
            current_epoch += 1
            train(train_dl, test_dl, model, loss_function, optimizer, scaler, device,
                  current_epoch, args, history)
    elif args.mode == 'test':
        validate(test_dl, model, loss_function, device, args)
    else:
        print('Invalid execution mode')


##########################
# Define the test function
##########################


@torch.no_grad()
def validate(dataloader, model, loss_function, device, args) -> RunningAverage:
    model.eval()

    batch_time = RunningAverage('BatchTime', ':6.3f')
    data_time = RunningAverage('DataTime', ':6.3f')
    loss_run_avg = RunningAverage('Loss', ':6.3f')

    timer = Timer(args.gpu)

    print('Testing...')

    for i, (masked_img_batch, mask_batch, gt_batch) in enumerate(dataloader):
        masked_img_batch = masked_img_batch.to(device)
        mask_batch = mask_batch.to(device)
        gt_batch = gt_batch.to(device)

        data_time.update(timer.elapsed())

        prediction, _ = model(masked_img_batch, mask_batch)
        loss_val = loss_function(gt_batch, mask_batch, prediction)

        loss_run_avg.update(loss_val.item(), masked_img_batch.size(0))

        batch_time.update(timer.elapsed())

        timer.reset()

        if (i + 1) * args.batch_size >= args.samples_per_validation:
            break

    print(f'[Validation batches: {len(dataloader)}] {loss_run_avg} | {data_time} | {batch_time})')

    return loss_run_avg


##############################
# Define the training function
##############################


def train(train_dl, test_dl, model, loss_function, optimizer, scaler, device, current_epoch, args, history):
    #torch.autograd.set_detect_anomaly(True)

    model.train()

    batch_time = RunningAverage('BatchTime', ':6.3f')
    data_time = RunningAverage('DataTime', ':6.3f')
    loss_run_avg = RunningAverage('Loss', ':6.3f')

    timer = Timer(args.gpu)

    print(f'Training epoch {current_epoch}')

    for i, (masked_img_batch, mask_batch, gt_batch) in enumerate(train_dl):
        masked_img_batch = masked_img_batch.to(device)
        mask_batch = mask_batch.to(device)
        gt_batch = gt_batch.to(device)

        data_time.update(timer.elapsed())

        with torch.cuda.amp.autocast(enabled=args.useAMP):
            prediction, _ = model(masked_img_batch, mask_batch)
            loss_val = loss_function(gt_batch, mask_batch, prediction)

        loss_run_avg.update(loss_val.item(), masked_img_batch.size(0))

        scaler.scale(loss_val).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        batch_time.update(timer.elapsed())

        if i % 10 == 0:
            mem = (torch.cuda.memory_allocated() + torch.cuda.memory_reserved()) // 1024**2
            print(f'[{i}/{len(train_dl)}] {loss_run_avg} | {data_time} | {batch_time} | Allocated memory: {mem} MBs')

        timer.reset()

        if (i + 1) * args.batch_size >= args.samples_per_epoch:
            break

    ##############################
    # Save checkpoint at epoch end
    ##############################

    with torch.no_grad():
        print('Recording loss history')
        val_loss_run_avg = validate(test_dl, model, loss_function, device, args)

        history.append({
            'train_loss': loss_run_avg.avg,
            'val_loss': val_loss_run_avg.avg,
            'finetuning': args.tuning_phase
        })
        loss_run_avg.reset()

        print('Saving checkpoint...')
        checkpoint_name = f'checkpoint.pth.tar'
        checkpoint_path = os.path.join(args.save_checkpoint_dir, checkpoint_name)
        torch.save({
            'current_epoch': current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'history': history
        }, checkpoint_path)

        if current_epoch % args.coarse_checkpoint_step == 0:
            coarse_checkpoint_path = os.path.join(args.save_checkpoint_dir, f'checkpoint_epoch_{current_epoch}.pth.tar')
            shutil.copy(checkpoint_path, coarse_checkpoint_path)


if __name__ == '__main__':
    main()

