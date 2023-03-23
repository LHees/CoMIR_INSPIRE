
### Written by Johan Ofverstedt
### Based on "https://devpress.csdn.net/python/630455ca7e6682346619a0f4.html"
###
### 2023

import torch

def rot90_3d_one(x, tind):
    if tind < 4:
        k = tind
        return torch.rot90(x, k, (-2, -1))
    elif tind < 8:
        k = tind-4
        x = torch.rot90(x, 2, (-3, -1))
        return torch.rot90(x, k, (-2, -1))
    elif tind < 12:
        k = tind-8
        x = torch.rot90(x, 1, (-3, -1))
        return torch.rot90(x, k, (-3, -2))
    elif tind < 16:
        k = tind-12
        x = torch.rot90(x, -1, (-3, -1))
        return torch.rot90(x, k, (-3, -2))
    elif tind < 20:
        k = tind-16
        x = torch.rot90(x, 1, (-3, -2))
        return torch.rot90(x, k, (-3, -1))
    elif tind < 24:
        k = tind-16
        x = torch.rot90(x, -1, (-3, -2))
        return torch.rot90(x, k, (-3, -1))
    else:
        assert(False)

def rot90_3d(x, tind):
    """ Function that applies rot90 3d transformations to a batch of images 'x'.
        tind can either be an integer, tensor containing a single integer, or 
        a tensor containing 'b' integers where 'b' is the same as the batch dimension
        (0) of 'x'. The valid range of the 'tind' values are [0, 23] for one
        of the 24 possible axis-aligned 3d rotations.
    """

    # both parameters are tensors
    if not torch.is_tensor(tind):
        tind = torch.tensor([tind], device=x.device, dtype=torch.int32)
        tind = tind.reshape(torch.prod(torch.tensor(tind.shape)))

    assert(torch.is_tensor(x))

    if tind.shape[0] == 1:
        # only a single transformation applied to all elements of the batch
        return rot90_3d_one(x, tind[0])
    elif tind.shape[0] == x.shape[0]:
        xs = []
        for i in range(x.shape[0]):
            xs.append(rot90_3d_one(x[i:i+1, ...], tind[i]))
        return torch.cat(xs, dim=0) 
    else:
        assert(False)






