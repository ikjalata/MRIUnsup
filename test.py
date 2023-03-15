import argparse
import os
from util import MRI_Dataset

import csv
import torch.nn as nn
import torch.utils.data
import torch
import torch.nn.parallel
import random
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML
from torch.utils.tensorboard import SummaryWriter
from model import Net
import util_torch as util_torch
import torch.optim.lr_scheduler as lr_scheduler
import losslayer
import metrics 

# set random seed
manualSeed = 999
print(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#set data and mask directories
dataroot = "/home/ibsa/medicalImaging/project_unsup2/data"
maskdir = "/home/ibsa/medicalImaging/ukashcode/MRRecon1/mask_ukash/masks_test_R2"

#set training parameters
workers = 2
batch_size = 1
ngpu = 2
K = 5
lmbda = 0
w_coeff = 0.002
tv_coeff = 0.005
checkpoint_path = "data_l1/ckpt/model_7.pt" #_gen_1599.pt"

#initialize the tensorboard writer
writer = SummaryWriter()

#define the function to initialize the network weights
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

#define the function to normalize the input tensor
def normalized(a):
    #xnormalized = a + ( ((x - xminimum) * (b - a)) / range of x)
    anormalized = (a - a.min())/(a.max()-a.min())
    #anormalized = (a + 255.0)/510.0
    return anormalized

# define the dataset and data loader
dataset = MRI_Dataset(data_dir = dataroot, mask_dir=maskdir,data_type="test", transform = transforms.ToTensor())
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
             shuffle=True, num_workers=workers)

#set the device to train on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu >1 )
         else "cpu")
print(device)

#initialize the network
net = Net(K, lmbda, device, n_hidden=128).to(device)

net.apply(weights_init)
print(net,"device.type::", device.type)


alpha = torch.tensor(tv_coeff).to(device)
beta = torch.tensor(w_coeff).to(device)
real_batch = next(iter(dataloader))


checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
#netG.load_state_dict(checkpoint)
#optimizerG.load_state_dict(checkpoint['optimizer_state_dict'])
#epoch = checkpoint['epoch']
#g_loss = checkpoint['loss']
net.eval()

img_list = []
G_losses = []
D_losses = []
iters = 0

inp_psnr = []
inp_nrmse = []
inp_ssim = []
out_psnr = []
out_nrmse = []
out_ssim = []

for i, data in enumerate(dataloader, 0):
    ksp_truth, ksp_input, truth_img, lowres_img, sensemp, masks = data
    
    ksp_truth = ksp_truth.to(device).permute(0,3,1,2)
    ksp_input = ksp_input.to(device).permute(0,3,1,2)
    
    sensemp = sensemp.to(device).permute(0,3,1,2)
    masks = masks.to(device)

    X_gen = net(ksp_input, sensemp)
    output_img = X_gen.permute(0, 2, 3, 1).detach().cpu()
    output_img = util_torch.channels_to_complex(output_img).numpy()
    truth_img = util_torch.channels_to_complex(truth_img).numpy()
    lowres_img = util_torch.channels_to_complex(lowres_img).numpy() 
    
    truth_img = abs(truth_img) #normalized(abs(truth_img))
    output_img = abs(output_img)
    lowres_img = abs(lowres_img) 

    psnr, nrmse, ssim = metrics.compute_all(truth_img, lowres_img)
    inp_psnr.append(psnr)
    inp_nrmse.append(nrmse)
    inp_ssim.append(ssim)

    pnr, nrms, ssm = metrics.compute_all(truth_img, output_img)
    out_psnr.append(pnr)
    out_nrmse.append(nrms)
    out_ssim.append(ssm)

    print("input psnr, nrmse, ssim:")
    print(
          np.round(np.mean(inp_psnr), decimals=3),
          np.round(np.mean(inp_nrmse), decimals=3),
          np.round(np.mean(inp_ssim), decimals=3)
            )
    print("output psnr, nrmse, ssim:")
    print(
          np.round(np.mean(out_psnr), decimals=3),
          np.round(np.mean(out_nrmse), decimals=3),
          np.round(np.mean(out_ssim), decimals=3)
            )
# save the performance result
with open('perfor_metrics/perf_l1/metrics_inp_r9.csv', 'w', newline='') as csvfile:
    fieldnames = ['input_psnr', 'input_nrmse', 'input_ssim'] 
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'input_psnr': inp_psnr})
    writer.writerow({'input_nrmse': inp_nrmse})
    writer.writerow({'input_ssim': inp_ssim}) 
with open('perfor_metrics/perf_l1/metrics_out_r9.csv', 'w', newline='') as csvfile:
    fieldnames = ['out_psnr', 'out_nrmse', 'out_ssim'] 
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

    writer.writeheader()
    writer.writerow({'out_psnr': out_psnr})
    writer.writerow({'out_nrmse': out_nrmse})
    writer.writerow({'out_ssim': out_ssim}) 















