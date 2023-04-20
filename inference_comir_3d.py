
#
# Script for performing inference of CoMIR:s
# Authors: Nicolas Pielawski, Elisabeth Wetzer, Johan Ofverstedt
# Published under the MIT License
# 2020
#

from models.tiramisu import DenseUNet
from train_comir import MultimodalDataset, ModNet, ModNet3D, ImgAugTransform3D

# Python Standard Libraries
from datetime import datetime
import glob
import itertools
import math
import os
import time
import sys
import random
import re
import warnings
from contextlib import nullcontext

# Deep Learning libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import torchvision

# Other libraries
# ~ Scientific
import numpy as np
import scipy
import scipy.stats as st
import scipy.special
# ~ Image manipulation
import imgaug
from imgaug import augmenters as iaa
import skimage
import skimage.io as skio
import skimage.transform as sktr
from tiler import Tiler, Merger

# Local libraries
from utils.image import *
from utils.torch import *

#logTransformA = True
#logTransformB = False

apply_sigmoid = True


def compute_padding(sz, alignment=128):
    if sz % alignment == 0:
        return 0
    else:
        return alignment - (sz % alignment)


def filenames_to_dict(filenamesA, filenamesB):
    d = {}
    for i in range(len(filenamesA)):
        basename = os.path.basename(filenamesA[i])
        d[basename] = (i, None)
    for i in range(len(filenamesB)):
        basename = os.path.basename(filenamesB[i])
        # filter out files only in B
        if basename in d:
            d[basename] = (d[basename][0], i)

    # filter out files only in A
    d = {k: v for k, v in d.items() if v[1] is not None}
    return d


if __name__ == "__main__":
    # %%
    print(len(sys.argv), sys.argv)
    if len(sys.argv) < 5:
        print('Use: inference_comir.py model_path mod_a_path mod_b_path mod_a_out_path mod_b_out_path')
        sys.exit(-1)

    model_path = sys.argv[1]
    modA_path = sys.argv[2]
    modB_path = sys.argv[3]
    modA_out_path = sys.argv[4]
    modB_out_path = sys.argv[5]

    if modA_path[-1] != '/':
        modA_path += '/'
    if modB_path[-1] != '/':
        modB_path += '/'
    if modA_out_path[-1] != '/':
        modA_out_path += '/'
    if modB_out_path[-1] != '/':
        modB_out_path += '/'

    if not os.path.exists(modA_out_path):
        os.makedirs(modA_out_path)
    if not os.path.exists(modB_out_path):
        os.makedirs(modB_out_path)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    checkpoint = torch.load(model_path, map_location=device)

    modelA = checkpoint['modelA']
    modelB = checkpoint['modelB']

    print("Loading dataset...")
    dset = MultimodalDataset(modA_path + '*', modB_path + '*', logA=modelA.log_transform, logB=modelB.log_transform, transform=None, dim=3)

    # Modality slicing
    # You can choose a set of channels per modality (RGB for instance)
    # Modality A
    modA_len = modelA.in_channels
    modA = slice(0, modA_len)
    modA_name = "A"
    # Modality B
    modB_len = modelB.in_channels
    modB = slice(modA_len, modA_len + modB_len)
    modB_name = "B"
    print('Modality A has ', modA_len, ' channels.', sep='')
    print('Modality B has ', modB_len, ' channels.', sep='')
    if modelA.log_transform:
        print('Modality A uses a log transform.')
    if modelB.log_transform:
        print('Modality B uses a log transform.')

    modelA.to(device)
    modelB.to(device)
    if torch.cuda.is_available():
        modelA.half()
        modelB.half()

    modelA.eval()
    modelB.eval()

    # Number of threads to use
    # It seems to be best at the number of physical cores when hyperthreading is enabled
    # In our case: 18 physical + 18 logical cores
    torch.set_num_threads(5)

    N = len(dset)
    idx = 1
    for i in range(N):
        print(f'Encoding... {idx}/{N}')
        item = dset.get(i, augment=False)

        autocast = device.type == 'cuda' and torch.__version__ >= '1.6.0'
        with torch.cuda.amp.autocast() if autocast else nullcontext():
            item = np.moveaxis(item, -1, 0)

            tiler = Tiler(data_shape=item.shape,
                          tile_shape=(item.shape[0], 64, 64, 64),
                          overlap=(0, 32, 32, 32),
                          channel_dimension=0)
            padded_shape, padding = tiler.calculate_padding()
            tiler.recalculate(data_shape=padded_shape)
            merger = Merger(tiler=tiler, window='hann')

            padded_item = np.pad(item, padding, mode='reflect')
            for tile_id, tile in tiler(padded_item, progress_bar=True):
                item = tile[np.newaxis, ...]  # add batch axis
                tensor = torch.tensor(item, device=device)
                if not autocast and device.type == 'cuda':
                    tensor = tensor.half()

                L1 = modelA(tensor[:, modA, ...])
                L2 = modelB(tensor[:, modB, ...])
                processed_tensor = torch.cat((L1, L2), dim=1)
                processed_tensor = torch.squeeze(processed_tensor, dim=0)  # remove batch axis

                processed_tile = processed_tensor.cpu().detach().numpy() if \
                    device.type == 'cuda' else processed_tensor.detach().numpy()
                merger.add(tile_id, processed_tile)

                del tile
                del L1
                del L2
                del processed_tensor

            processed_item = merger.merge(extra_padding=padding)
            processed_item = np.moveaxis(processed_item, 0, -1)
            if apply_sigmoid:
                processed_item = np.round(scipy.special.expit(processed_item) * 255).astype('uint8')
            im1 = processed_item[..., modA]
            im2 = processed_item[..., modB]

            path1 = modA_out_path + dset.get_name(i)
            path2 = modB_out_path + dset.get_name(i)
            skio.imsave(path1, im1, plugin='simpleitk')
            skio.imsave(path2, im2, plugin='simpleitk')
            idx += 1
