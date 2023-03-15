#the code partial borrowed from
# "Neural Network-based Reconstruction in Compressed Sensing
#MRI Without Fully-sampled Training Data"

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import util_torch as util_torch
def absval(arr):
    """
    Takes absolute value of last dimension, if complex.
    Input dims:  (N, l, w, 2)
    Output dims: (N, l, w)
    """
    # Expects input of size (N, l, w, 2)
    assert arr.shape[-1] == 2
    return torch.norm(arr, dim=3)
    
def scale(y, y_zf):
    """Scales inputs for numerical stability"""
    flat_yzf = torch.flatten(absval(y_zf), start_dim=1, end_dim=2)
    max_val_per_batch, _ = torch.max(flat_yzf, dim=1, keepdim=True)
    y = y / max_val_per_batch.view(len(y), 1, 1, 1)
    y_zf = y_zf / max_val_per_batch.view(len(y), 1, 1, 1)
    return y, y_zf

class Upsample(nn.Module):
    """Upsamples input multi-channel image"""
    def __init__(self, scale_factor, mode, align_corners):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners
    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=self.align_corners)

class ResBlock(nn.Module):
    '''5-layer CNN with residual output'''
    def __init__(self, n_ch_in=2, n_ch_out=2, nf=64, ks=3):
        
        super(ResBlock, self).__init__()
        self.n_ch_out = n_ch_out

        self.conv1 = nn.Conv2d(n_ch_in, nf, ks, padding = ks//2)
        self.conv2 = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv3 = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv4 = nn.Conv2d(nf, nf, ks, padding = ks//2)
        self.conv5 = nn.Conv2d(nf, n_ch_out, ks, padding = ks//2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv1_out = self.relu(conv1_out)

        conv2_out = self.conv2(conv1_out)
        conv2_out = self.relu(conv2_out)

        conv3_out = self.conv3(conv2_out)
        conv3_out = self.relu(conv3_out)

        conv4_out = self.conv4(conv3_out)
        conv4_out = self.relu(conv4_out)

        conv5_out = self.conv5(conv4_out)

        x_res = x[:,:self.n_ch_out,:,:] + conv5_out
        return x_res


class Net(nn.Module):
    def __init__(self, K, lmbda, device, n_hidden=64):
        super(Net, self).__init__()

        #self.mask = mask
        self.lmbda = lmbda
        self.resblocks = nn.ModuleList()
        self.device = device
            
        for i in range(K):
            resblock = ResBlock(n_ch_in=2, nf=n_hidden)
            self.resblocks.append(resblock)

        self.block_final = ResBlock(n_ch_in=2, nf=n_hidden)


    def forward(self, ksp_input, sensemap, window = 1, mask = None):
        
        if mask is None:
           mask=torch.not_equal(ksp_input, 0)
           dtype=torch.complex64 
           mask = mask.type(dtype)
        x = util_torch.transpose_model(ksp_input * window, sensemap)
        x = util_torch.complex_to_channels(x)#;print(x.shape);quit()
        #ksp_input, x = scale(ksp_input, x)
        for i in range(len(self.resblocks)):
            # z-minimization
            x = x.permute(0, 3, 1, 2)
            
            z = self.resblocks[i](x)
            z = z.permute(0, 2, 3, 1)
            z = util_torch.channels_to_complex(z) 
            # x-minimization
            #z_ksp = utils.fft(z)
            z_ksp = util_torch.model_forward(z, sensemap)
            #x_ksp = losslayer.data_consistency(z_ksp, y, self.mask, self.lmbda)
            x_ksp = (1 - mask) * z_ksp + mask * (self.lmbda*z_ksp + ksp_input) / (1 + self.lmbda)
            #x = utils.ifft(x_ksp)
            x = util_torch.transpose_model(x_ksp, sensemap)
            x = util_torch.complex_to_channels(x)
        x = x.permute(0, 3, 1, 2)
        x = self.block_final(x)
        return x
