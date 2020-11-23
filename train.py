import argparse
import logging
import os
import sys
import numpy as np
from tqdm import tqdm
import time

import torch
import torch.nn as nn
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from models.unet import UNet
from models.nested_unet import NestedUNet

from datasets.promise12 import Promise12
from datasets.chaos import Chaos

from dice_loss import DiceBCELoss, dice_coeff
from eval import eval_net


torch.manual_seed(2020)

def train_net(net, trainset, valset, device, epochs, batch_size, lr, weight_decay, log_save_path):

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    writer = SummaryWriter(log_dir=log_save_path)

    optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.95)
    criterion = DiceBCELoss()

    best_DSC = 0.0
    for epoch in range(epochs):
        logging.info(f'Epoch {epoch + 1}')
        epoch_loss = 0
        epoch_dice = 0
        with tqdm(total=len(trainset), desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                net.train()
                imgs = batch['image']
                true_masks = batch['mask']
        
                imgs = imgs.to(device=device, dtype=torch.float32)
                true_masks = true_masks.to(device=device, dtype=torch.float32)
                masks_pred = net(imgs)

                pred = torch.sigmoid(masks_pred)
                pred = (pred>0.5).float()
                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                epoch_dice += dice_coeff(pred, true_masks).item()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 5)
                optimizer.step()

                pbar.set_postfix(**{'loss (batch)': loss.item()})
                pbar.update(imgs.shape[0])

        scheduler.step()

        logging.info('Training loss:   {}'.format(epoch_loss/len(train_loader)))
        writer.add_scalar('Train/loss', epoch_loss/len(train_loader), epoch)  
        logging.info('Training DSC:    {}'.format(epoch_dice/len(train_loader)))
        writer.add_scalar('Train/dice', epoch_dice/len(train_loader), epoch)     

        val_dice, val_loss = eval_net(net, val_loader, device, criterion)  
        logging.info('Validation Loss: {}'.format(val_loss))
        writer.add_scalar('Val/loss', val_loss, epoch)
        logging.info('Validation DSC:  {}'.format(val_dice))
        writer.add_scalar('Val/dice', val_dice, epoch)

        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # writer.add_images('images', imgs, epoch)
        writer.add_images('masks/true', true_masks, epoch)
        writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, epoch)

    writer.close()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch_size', metavar='B', type=int, nargs='?', default=8, help='Batch size')
    parser.add_argument('--lr', metavar='LR', type=float, nargs='?', default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, nargs='?', default=1e-5, help='Weight decay')
    parser.add_argument('--model', type=str, default='unet', help='Model name')
    parser.add_argument('--dataset', type=str, default='promise12', help='Dataset name')
    parser.add_argument('--gpu', type=int, default='0', help='GPU number')
    parser.add_argument('--save', type=str, default='EXP', help='Experiment name')
    return parser.parse_args()


if __name__ == '__main__':

    args = get_args()
    args.save = 'logs_train/{}-{}-{}'.format(args.model, args.dataset, time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)
    logging.info(f''' 
        Model:             {args.model}
        Dataset:           {args.dataset}
        Total Epochs:      {args.epochs}
        Batch size:        {args.batch_size}
        Learning rate:     {args.lr}
        Weight decay:      {args.weight_decay}
        Device:            GPU{args.gpu}
        Log name:          {args.save}
    ''')

    torch.cuda.set_device(args.gpu)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # choose a model
    if args.model == 'unet':
        net = UNet()
    elif args.model == 'nestedunet':
        net = NestedUNet()
   
    net.to(device=device)

    
    # choose a dataset
    
    if args.dataset == 'promise12':
        dir_data = '../data/promise12'
        trainset = Promise12(dir_data, mode='train')
        valset = Promise12(dir_data, mode='val')
    elif args.dataset == 'chaos':
        dir_data = '../data/chaos'
        trainset = Chaos(dir_data, mode='train')
        valset = Chaos(dir_data, mode='val')
    
    try:
        train_net(net=net,
                  trainset=trainset,
                  valset=valset,
                  epochs=args.epochs,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  weight_decay=args.weight_decay,
                  device=device,
                  log_save_path=args.save)
    except KeyboardInterrupt:
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
