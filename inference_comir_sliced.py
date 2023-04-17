
#
# Script for performing inference of CoMIR:s
# Authors: Nicolas Pielawski, Elisabeth Wetzer, Johan Ofverstedt
# Published under the MIT License
# 2020
#

from models.tiramisu import DenseUNet
from train_comir import MultimodalDataset, ModNet, ImgAugTransform

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
    if len(sys.argv) < 6:
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

    from train_comir import ModNet3D  # TODO: remove
    checkpoint = torch.load(model_path, map_location=device)

    modelA = checkpoint['modelA']
    modelB = checkpoint['modelB']

    print("Loading dataset...")
    dset = MultimodalDataset(modA_path + '*', modB_path + '*', logA=modelA.log_transform, logB=modelB.log_transform, transform=None, dim=2)

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
    subject = dset.get_name(0)[:8]
    comir_arr = None
    print('Encoding... 1/?')  # TODO: find out how many subjects there are ahead of time
    for i in range(N):
        item = dset.get(i, augment=False)

        if dset.get_name(i)[:8] != subject:  # finished inference for a subject
            im1 = comir_arr[..., modA]
            im2 = comir_arr[..., modB]
            skio.imsave(modA_out_path + subject + ".nii.gz", im1, plugin='simpleitk')
            skio.imsave(modB_out_path + subject + ".nii.gz", im2, plugin='simpleitk')
            subject = dset.get_name(i)[:8]
            comir_arr = None
            idx += 1
            print(f'Encoding... {idx}/?')

        autocast = device.type == 'cuda' and torch.__version__ >= '1.6.0'
        with torch.cuda.amp.autocast() if autocast else nullcontext():
            item = item[np.newaxis, ...]  # add batch axis (doubles as depth axis)
            item = torch.tensor(np.stack(item), device=device)
            item = item.movedim(-1, 1)

            if not autocast and device.type == 'cuda':
                item = item.half()

            padsz = 128
            orig_shape = item.shape
            pad1 = compute_padding(item.shape[-1], alignment=padsz)
            pad2 = compute_padding(item.shape[-2], alignment=padsz)

            padded_item = F.pad(item, (padsz, padsz + pad1, padsz, padsz + pad2),
                                mode='reflect')
            L1 = modelA(padded_item[:, modA])
            L2 = modelB(padded_item[:, modB])

            L1 = L1[:, :, padsz:padsz+orig_shape[2], padsz:padsz+orig_shape[3]]
            L2 = L2[:, :, padsz:padsz+orig_shape[2], padsz:padsz+orig_shape[3]]

            rep1 = L1.movedim(0, -1)
            rep2 = L2.movedim(0, -1)
            if device.type == 'cuda':
                im1 = rep1.cpu().detach().numpy()
                im2 = rep2.cpu().detach().numpy()
            else:
                im1 = rep1.detach().numpy()
                im2 = rep2.detach().numpy()
            if apply_sigmoid:
                im1 = np.round(scipy.special.expit(im1) * 255).astype('uint8')
                im2 = np.round(scipy.special.expit(im2) * 255).astype('uint8')

            if comir_arr is None:
                comir_arr = np.append(im1, im2, axis=-1)
            else:
                comir_arr = np.vstack((comir_arr, np.append(im1, im2, axis=-1)))

            del item
            del padded_item
            del L1
            del L2
            del rep1
            del rep2

    # Save the last CoMIR
    im1 = comir_arr[..., modA]
    im2 = comir_arr[..., modB]
    skio.imsave(modA_out_path + subject + ".nii.gz", im1, plugin='simpleitk')
    skio.imsave(modB_out_path + subject + ".nii.gz", im2, plugin='simpleitk')
