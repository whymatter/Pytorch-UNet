import argparse
import logging
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

import wandb
from evaluate import evaluate
from unet import UNet
from utils.data_loading import BasicDataset

dir_img = Path('./data/imgs/')
dir_mask = Path('./data/masks/')
dir_checkpoint = Path('./checkpoints/')


def train_model(
        model,
        device,
        batch_size: int = 1,
        val_percent: float = 0.1,
        img_scale: float = 0.5,
        amp: bool = False,
):
    # 1. Create dataset
    dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    _, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

    # (Initialize logging)
    experiment = wandb.init(id="am3jqymn", resume="allow")

    # 5. Begin training
    checkpoints = os.listdir('./checkpoints')
    for epoch in tqdm(range(1, len(checkpoints) + 1), unit='epoch'):
        checkpoint = f'./checkpoints/checkpoint_epoch{epoch}.pth'

        # Load checkpoint
        state_dict = torch.load(checkpoint, map_location=device)
        del state_dict['mask_values']
        model.load_state_dict(state_dict)

        # Run validation
        val_score = evaluate(model, val_loader, device, amp)

        # Log validation
        logging.info('Validation Dice score: {}'.format(val_score))
        # experiment.log({
        #     'validation Dice': val_score,
        #     'epoch': epoch
        # })


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--scale', '-s', type=float, default=0.5, help='Downscaling factor of the images')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=1, help='Batch size')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--bilinear', action='store_true', default=False, help='Use bilinear upsampling')
    parser.add_argument('--classes', '-c', type=int, default=2, help='Number of classes')
    parser.add_argument('--channels', '-ch', type=int, default=3, help='Number of channels')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=args.channels, n_classes=args.classes, bilinear=args.bilinear)
    model = model.to(memory_format=torch.channels_last)

    model.to(device=device)
    
    train_model(
        model=model,
        device=device,
        batch_size=args.batch_size,
        val_percent=args.val / 100,
        img_scale=args.scale,
        amp=args.amp
    )