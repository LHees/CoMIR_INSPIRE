
#
# Script for performing inference of CoMIR:s
# Authors: Nicolas Pielawski, Elisabeth Wetzer, Johan Ofverstedt
# Published under the MIT License
# 2020
#

from models.tiramisu import DenseUNet
from train_comir import MultimodalDataset, ModNet, ModNet3D

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
    d = {k:v for k,v in d.items() if v[1] is not None}
    return d


if __name__ == "__main__":
    # %%
    print(len(sys.argv), sys.argv)
    if len(sys.argv) < 6:
        print('Use: inference_comir.py model_path mod_a_path mod_b_path mod_a_out_path mod_b_out_path [spat_dim]')
        sys.exit(-1)

    model_path = sys.argv[1]
    modA_path = sys.argv[2]
    modB_path = sys.argv[3]
    modA_out_path = sys.argv[4]
    modB_out_path = sys.argv[5]
    dim = int(sys.argv[6]) if len(sys.argv) >= 7 else 2

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
    dset = MultimodalDataset(modA_path + '*', modB_path + '*', logA=modelA.log_transform, logB=modelB.log_transform, transform=None, dim=dim)

    # Modality slicing
    # You can choose a set of channels per modality (RGB for instance)
    # Modality A
    modA_len = modelA.in_channels #dset.channels[0]
    modA = slice(0, modA_len)
    modA_name = "A"
    # Modality B
    modB_len = modelB.in_channels #dset.channels[1]
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

    # How many images to compute in one iteration?
    batch_size = 1

    N = len(dset)
    l, r = 0, batch_size
    idx = 1
    for i in range(int(np.ceil(N / batch_size))):
        batch = []
        names = []
        for j in range(l, r):
            batch.append(dset.get(j, augment=False))
            names.append(dset.get_name(j))

        autocast = device.type == 'cuda' and torch.__version__ >= '1.6.0'
        with torch.cuda.amp.autocast() if autocast else nullcontext():
            batch = torch.tensor(np.stack(batch), device=device)
            batch = batch.permute(0, -1, 1, 2, 3) if dim == 3 else batch.permute(0, -1, 1, 2)

            if not autocast and device.type == 'cuda':
                batch = batch.half()

            padsz = 8 if dim == 3 else 128
            orig_shape = batch.shape
            pad1 = compute_padding(batch.shape[-1])
            pad2 = compute_padding(batch.shape[-2])
            if dim == 3:
                pad3 = compute_padding(batch.shape[-3])

            padded_batch = F.pad(batch, (padsz, padsz+pad1, padsz, padsz+pad2, padsz, padsz+pad3) if dim == 3 else
                                        (padsz, padsz+pad1, padsz, padsz+pad2), mode='reflect')
            L1 = modelA(padded_batch[:, modA])
            L2 = modelB(padded_batch[:, modB])

            L1 = L1[:, :, padsz:padsz+orig_shape[2], padsz:padsz+orig_shape[3], padsz:padsz+orig_shape[4]] \
                if dim == 3 else L1[:, :, padsz:padsz+orig_shape[2], padsz:padsz+orig_shape[3]]
            L2 = L2[:, :, padsz:padsz+orig_shape[2], padsz:padsz+orig_shape[3], padsz:padsz+orig_shape[4]] \
                if dim == 3 else L2[:, :, padsz:padsz+orig_shape[2], padsz:padsz+orig_shape[3]]

            for j in range(len(batch)):
                path1 = modA_out_path + names[j]
                path2 = modB_out_path + names[j]

                rep1 = L1[j].permute(1, 2, 3, 0) if dim == 3 else L1[j].permute(1, 2, 0)
                rep2 = L2[j].permute(1, 2, 3, 0) if dim == 3 else L2[j].permute(1, 2, 0)
                if device.type == 'cuda':
                    im1 = rep1.cpu().detach().numpy()
                    im2 = rep2.cpu().detach().numpy()
                else:
                    im1 = rep1.detach().numpy()
                    im2 = rep2.detach().numpy()
                if apply_sigmoid:
                    im1 = np.round(scipy.special.expit(im1) * 255).astype('uint8')
                    im2 = np.round(scipy.special.expit(im2) * 255).astype('uint8')

                skio.imsave(path1, im1, plugin='simpleitk' if dim == 3 else None)
                skio.imsave(path2, im2, plugin='simpleitk' if dim == 3 else None)
                print(f'Encodeing... {idx}/{N}')
                idx += 1






    #     if device.type == 'cuda' and torch.__version__ >= '1.6.0':
    #         with torch.cuda.amp.autocast(): # pytorch>=1.6.0 required
    #             batch = torch.tensor(np.stack(batch), device=device)
    #             batch = batch.permute(0, -1, 1, 2, 3) if dim == 3 else batch.permute(0, -1, 1, 2)
    #
    # #            if device.type == 'cuda' and not use_AMP:
    # #                batch = batch.half()
    #
    #             padsz = 8 if dim == 3 else 128
    #             orig_shape = batch.shape
    #             pad1 = compute_padding(batch.shape[-1])
    #             pad2 = compute_padding(batch.shape[-2])
    #             if dim == 3:
    #                 pad3 = compute_padding(batch.shape[-3])
    #
    #             padded_batch = F.pad(batch, (padsz, padsz+pad1, padsz, padsz+pad2, padsz, padsz+pad3) if dim == 3 else
    #                                         (padsz, padsz+pad1, padsz, padsz+pad2), mode='reflect')
    #             #newdim = (np.array(batch.shape[2:]) // 128) * 128
    #             #L1 = modelA(batch[:, modA, :newdim[0], :newdim[1]])
    #             #L2 = modelB(batch[:, modB, :newdim[0], :newdim[1]])
    #             L1 = modelA(padded_batch[:, modA])
    #             L2 = modelB(padded_batch[:, modB])
    #
    #             L1 = L1[:, :, padsz:padsz+orig_shape[2], padsz:padsz+orig_shape[3], padsz:padsz+orig_shape[4]] \
    #                 if dim == 3 else L1[:, :, padsz:padsz+orig_shape[2], padsz:padsz+orig_shape[3]]
    #             L2 = L2[:, :, padsz:padsz+orig_shape[2], padsz:padsz+orig_shape[3], padsz:padsz+orig_shape[4]] \
    #                 if dim == 3 else L2[:, :, padsz:padsz+orig_shape[2], padsz:padsz+orig_shape[3]]
    #
    #             for j in range(len(batch)):#L1.shape[0]):
    #                 # TODO: modify for 3D
    #                 path1 = modA_out_path + names[j]
    #                 path2 = modB_out_path + names[j]
    #                 im1 = L1[j].permute(1, 2, 0).cpu().detach().numpy()
    #                 im2 = L2[j].permute(1, 2, 0).cpu().detach().numpy()
    #
    #                 if apply_sigmoid:
    #                     im1 = np.round(scipy.special.expit(im1) * 255).astype('uint8')
    #                     im2 = np.round(scipy.special.expit(im2) * 255).astype('uint8')
    #                     skio.imsave(path1, im1)
    #                     skio.imsave(path2, im2)
    #                 else:
    #                     skio.imsave(path1, im1)
    #                     skio.imsave(path2, im2)
    #                 print(f'Encodeing... {idx}/{N}')
    #                 idx += 1
    #     else:
    #         batch = torch.tensor(np.stack(batch), device=device)
    #         batch = batch.permute(0, -1, 1, 2, 3) if dim == 3 else batch.permute(0, -1, 1, 2)
    #
    #         if device.type == 'cuda':
    #             batch = batch.half()
    #
    #         padsz = 8 if dim == 3 else 128
    #         orig_shape = batch.shape
    #         pad1 = compute_padding(batch.shape[-1])
    #         pad2 = compute_padding(batch.shape[-2])
    #         if dim == 3:
    #             pad3 = compute_padding(batch.shape[-3])
    #
    #         padded_batch = F.pad(batch, (padsz, padsz+pad1, padsz, padsz+pad2, padsz, padsz+pad3) if dim == 3 else
    #                                     (padsz, padsz+pad1, padsz, padsz+pad2), mode='reflect')
    #         #newdim = (np.array(batch.shape[2:]) // 128) * 128
    #         #L1 = modelA(batch[:, modA, :newdim[0], :newdim[1]])
    #         #L2 = modelB(batch[:, modB, :newdim[0], :newdim[1]])
    #         L1 = modelA(padded_batch[:, modA])
    #         L2 = modelB(padded_batch[:, modB])
    #
    #         L1 = L1[:, :, padsz:padsz+orig_shape[2], padsz:padsz+orig_shape[3], padsz:padsz+orig_shape[4]] \
    #             if dim == 3 else L1[:, :, padsz:padsz+orig_shape[2], padsz:padsz+orig_shape[3]]
    #         L2 = L2[:, :, padsz:padsz+orig_shape[2], padsz:padsz+orig_shape[3], padsz:padsz+orig_shape[4]] \
    #             if dim == 3 else L2[:, :, padsz:padsz+orig_shape[2], padsz:padsz+orig_shape[3]]
    #
    #         for j in range(len(batch)):#L1.shape[0]):
    #             # TODO: modify for 3D
    #             path1 = modA_out_path + names[j]
    #             path2 = modB_out_path + names[j]
    #
    #     #        print(path1)
    #     #        print(path2)
    #
    #     #        all_paths.append(path1)
    #     #        all_paths.append(path2)
    #
    #             if device.type == 'cuda':
    #                 im1 = L1[j].permute(1, 2, 0).cpu().detach().numpy()
    #                 im2 = L2[j].permute(1, 2, 0).cpu().detach().numpy()
    #             else:
    #                 im1 = L1[j].permute(1, 2, 0).detach().numpy()
    #                 im2 = L2[j].permute(1, 2, 0).detach().numpy()
    #             if apply_sigmoid:
    #                 im1 = np.round(scipy.special.expit(im1) * 255).astype('uint8')
    #                 im2 = np.round(scipy.special.expit(im2) * 255).astype('uint8')
    #                 skio.imsave(path1, im1)
    #                 skio.imsave(path2, im2)
    #             else:
    #                 skio.imsave(path1, im1)
    #                 skio.imsave(path2, im2)
    #             print(f'Encodeing... {idx}/{N}')
    #             idx += 1

        del L1
        del L2

        l, r = l+batch_size, r+batch_size
        if r > N:
            r = N
