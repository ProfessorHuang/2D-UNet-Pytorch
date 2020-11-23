import torch
import torch.nn.functional as F
from tqdm import tqdm

from dice_loss import dice_coeff, DiceBCELoss


def eval_net(net, loader, device, criterion):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    batch_size = loader.batch_size
    n_val = len(loader) * batch_size  # the number of batch
    tot_dice = 0
    tot_loss = 0

    with tqdm(total=n_val, desc='Validation round', unit='img', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                masks_pred = net(imgs)
                loss = criterion(masks_pred, true_masks)
                tot_loss += loss.item()
            
                pred = torch.sigmoid(masks_pred)
                pred = (pred > 0.5).float()
                tot_dice += dice_coeff(pred, true_masks).item()
            pbar.update(batch_size)

    net.train()
    return tot_dice / len(loader), tot_loss / len(loader)
