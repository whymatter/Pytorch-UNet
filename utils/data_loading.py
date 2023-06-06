import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from os import listdir
from os.path import splitext, isfile, join
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    mask_file = list(mask_dir.glob(idx + mask_suffix + '.*'))[0]
    mask = np.asarray(load_image(mask_file))
    if mask.ndim == 2:
        return np.unique(mask)
    elif mask.ndim == 3:
        mask = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim}')


class CachedDatset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', num_images: int = None):
        base = BasicDataset(images_dir, mask_dir, scale, mask_suffix, num_images)
        num_images = len(base)

        self.mask_values = base.mask_values

        self.img_cache = None
        self.mask_cache = None

        logging.info('Populating cache')

        for i, name in tqdm(enumerate(base.ids), total=num_images, desc='Loading images'):
            mask_file = mask_dir / (name + '.bmp')
            img_file = images_dir / (name + '.png')

            img = load_image(img_file)
            mask = load_image(mask_file)

            assert img.size == mask.size, \
                f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

            img = self.preprocess(base.mask_values, img, scale, is_mask=False)
            mask = self.preprocess(base.mask_values, mask, scale, is_mask=True)

            if self.img_cache is None:
                # initialize cache
                self.img_cache = np.zeros((num_images, *img.shape), dtype=np.uint8)
                self.mask_cache = np.zeros((num_images, *mask.shape), dtype=np.uint8)

            self.img_cache[i,:,:] = img
            self.mask_cache[i,:,:] = mask

        logging.info(f'Loaded image cache, size: {self.img_cache.nbytes}bytes, {self.img_cache.nbytes // 1024  // 1024  // 1024}Gbytes')
        logging.info(f'Loaded masks cache, size: {self.mask_cache.nbytes}bytes, {self.mask_cache.nbytes // 1024  // 1024  // 1024}Gbytes')


    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        img = np.asarray(pil_img, dtype=np.uint8)

        if is_mask:
            mask = np.zeros((h, w), dtype=np.uint8)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            return img


    def __len__(self):
        return self.img_cache.shape[0]

    def __getitem__(self, idx):
        assert idx < len(self), f'Requested index {idx} is out of range'
        img, mask = self.img_cache[idx,:,:], self.mask_cache[idx,:,:]
        img = img / 255.0
        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).to(torch.uint8).contiguous()
        }


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = '', num_images: int = None):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix
        self.num_images = num_images

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if isfile(join(images_dir, file)) and not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        
        if self.num_images is not None:
            self.ids = list(self.ids[:self.num_images])

        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')

        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))

        self.mask_values = list(sorted(np.unique(np.concatenate(unique), axis=0).tolist()))
        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]

        mask_file = self.mask_dir / (name + '.bmp')
        img_file = self.images_dir / (name + '.png')

        mask = load_image(mask_file)
        img = load_image(img_file)

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask', num_images=None)
