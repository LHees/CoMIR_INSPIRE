import torch
import numpy as np
import torch.nn.functional as F


class OverSampler(torch.utils.data.Sampler):
    def __init__(self, data_source, steps_per_epoch):
        self.data_source = data_source
        self.steps_per_epoch = steps_per_epoch

        if not isinstance(steps_per_epoch, int) or self.steps_per_epoch <= 0:
            raise ValueError("steps_per_epoch should be a positive integer "
                             "value, but got num_samples={}".format(self.steps_per_epoch))

    @property
    def num_samples(self):
        return self.steps_per_epoch

    def __iter__(self):
        n = len(self.data_source)
        return iter(np.random.choice(np.arange(n), self.steps_per_epoch, replace=True))

    def __len__(self):
        return self.steps_per_epoch


def rigid_transform(img, x, y, angle, mode="bilinear", device="cuda"):
    """Applies a rigid transformation to an image.
    Args:
        img (Tensor): a tensor of the shape (N, C, H, W).
        x (list of int): A list of positions [px] of size N to shift the image (x-axis).
        y (list of int): A list of positions [px] of size N to shift the image (y-axis).
        angle (list of int): A list of angles [degrees] of size N to rotate the image.
    """
    assert img.shape[0] == len(x) == len(y) == len(angle)
    angle = np.array(angle) / 180. * np.pi
    theta = np.empty((len(angle), 2, 3))
    theta[:, :, :2] = np.tile(angle, (2, 2, 1)).T
    theta[:, 0, 0] =  np.cos(theta[:, 0, 0])
    theta[:, 0, 1] = -np.sin(theta[:, 0, 1])
    theta[:, 1, 0] =  np.sin(theta[:, 1, 0])
    theta[:, 1, 1] =  np.cos(theta[:, 1, 1])
    theta[:, 0, 2] = -2*np.array(x)/img.shape[2]
    theta[:, 1, 2] = -2*np.array(y)/img.shape[3]
    theta = torch.tensor(theta, dtype=torch.float).to(device)
    grid = F.affine_grid(theta, img.shape, align_corners=True)
    return F.grid_sample(img, grid, mode, align_corners=True)


def activation_decay(tensors, p=2., device=None):
    """Computes the L_p^p norm over an activation map.
    """
    if not isinstance(tensors, list):
        tensors = [tensors]
    loss = torch.tensor(1.0, device=device)
    Z = 0
    for tensor in tensors:
        Z += tensor.numel()
        loss += torch.sum(tensor.pow(p).abs()).to(device)
    return loss / Z


def batch_rotate_p4(batch, k, device=None):  # TODO: rename?
    """Rotates each sample in a batch.
    Args:
        batch (Tensor): the batch to rotate, format is (N, C, [D], H, W).
        k (ndarray of int): the rotations to perform for each sample:
            k[i]*90 degrees for 2D images, or
            sequence number k[i] for 3D images (see https://stackoverflow.com/a/50546727).
        device (str): the device to allocate memory and run computations on.
    
    Returns (Tensor):
        The rotated batch.
    """
    batch_size = batch.shape[0]
    assert len(k) == batch_size, "The size of k must be equal to the batch size."
    cube_rotations = ['', 'X', 'Y', 'Z', 'XX', 'XY', 'XZ', 'YX', 'YY', 'ZY',
                      'ZZ', 'XXX', 'XXY', 'XXZ', 'XYX', 'XYY', 'XZZ', 'YXX',
                      'YYY', 'ZZZ', 'XXXY', 'XXYX', 'XYXX', 'XYYY']
    batch_p4 = torch.clone(batch).to(device)
    for i in range(batch_size):
        if batch.dim() == 5:  # 3D images
            for rot in cube_rotations[k[i]][::-1]:  # reverse the string
                dims = (1, 2) if rot == 'X' else (1, 3) if rot == 'Y' else (2, 3)
                batch_p4[i] = torch.rot90(batch_p4[i], dims=dims)
        else:  # 2D images assumed
            batch_p4[i] = torch.rot90(batch_p4[i], k=int(k[i]), dims=(1, 2))
    return batch_p4


def batch_affine(batch, transform, device=None):
    """Applies each transform in transforms to the corresponding image in each batch


    """
    
    batch_size = batch.shape[0]
    batch_affine = torch.empty_like(batch).to(device)
    for i in range(batch_size):
        
        batch_affine[i] =transform(batch[i])#affine(batch[i], float(angle[i]), [0, 0], float(scale[i]), [float(shear_x[i]), float(shear_y[i])])
        #batch_affine[i] = transforms[i](batch[i])
    return batch_affine


def batch_displace(batch, transform, device=None):
    batch_size = batch.shape[0]
    channels = batch.shape[1]
    batch_displaced = torch.empty_like(batch).to(device)
    #batch_displaced = transform(batch)
    for i in range(batch_size):
        batch_displaced[i,:,:,:] = transform[0,:,:,:](batch[i,:,:,:])
    return batch_displaced


def batch_displace_affine(batch, transform, device=None):
    displace_transform, affine_transform = transform
    batch = batch_displace(batch, displace_transform, device)
    batch = batch_affine(batch, affine_transform, device)
    return batch
