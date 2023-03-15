import argparse
import os
from util import MRI_Dataset

import torch.nn as nn
import torch.utils.data 
import torch
import torch.nn.parallel
import random 
import torch.backends.cudnn as cudnn 
import torch.optim as optim 
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

# set random seed
manualSeed = 999
print(manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

#set data and mask directories
dataroot = "/home/ibsa/medicalImaging/project_unsup2/data"
maskdir = "/home/ibsa/medicalImaging/ukashcode/MRRecon1/mask_ukash/masks"

#set training parameters
workers = 2
batch_size = 16
num_epochs = 50
lr = 0.001
beta1 = 0.99
ngpu = 2
K = 5
lmbda = 0
w_coeff = 0.002 
tv_coeff = 0.005
checkpoint_path = "ckpt"

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
dataset = MRI_Dataset(data_dir = dataroot, mask_dir=maskdir,data_type="train", transform = transforms.ToTensor()) 
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
             shuffle=True, num_workers=workers)

#set the device to train on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu >1 )
         else "cpu")
print(device)

real_batch = next(iter(dataloader))


#initialize the network
net = Net(K, lmbda, device, n_hidden=128).to(device)

net.apply(weights_init)
print(net,"device.type::", device.type)


#initialize the optimizer
optimizer = optim.Adam(net.parameters(), lr=lr, betas=(beta1, 0.999))


iters = 0
scheduler = lr_scheduler.StepLR(optimizer, step_size = 2, gamma=0.1)
alpha = torch.tensor(tv_coeff).to(device)
beta = torch.tensor(w_coeff).to(device)
for epoch in range(num_epochs):
    for i, data in enumerate(dataloader, 0):
        ksp_truth, ksp_input, truth_img, lowres_img, sensemp, masks = data
        
        ksp_truth = ksp_truth.to(device).permute(0,3,1,2)
        ksp_input = ksp_input.to(device).permute(0,3,1,2)
        truth_img = truth_img.to(device)
        lowres_img = lowres_img.to(device)
        sensemp = sensemp.to(device).permute(0,3,1,2)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        with torch.set_grad_enabled('train' == 'train'):  
             Y_real = truth_img.permute(0, 3, 1, 2)
             X_gen = net(ksp_input, sensemp)
             loss = losslayer.unsup_loss(X_gen, ksp_input, sensemp, alpha, beta, device)
             print("loss====",loss)
             loss.backward()
             optimizer.step()
        
        
        if (iters % 50 == 0):
            writer.add_scalar("Loss/loss", loss, iters)
            with torch.no_grad():
                output = net(ksp_input, sensemp)
                output = output.detach().cpu()
                
                output = output.permute(0, 2,3, 1);
                
        
                output = util_torch.channels_to_complex(output)
                truth_inp = util_torch.channels_to_complex(truth_img)
                lowres_inp = util_torch.channels_to_complex(lowres_img) 
               
                              
                output = (torch.abs(output)-torch.abs(output).min())/(torch.abs(output).max()-torch.abs(output).min())
                truth_inp = (torch.abs(truth_inp)-torch.abs(truth_inp).min())/(torch.abs(truth_inp).max()-torch.abs(truth_inp).min())
                lowres_inp = (torch.abs(lowres_inp)-torch.abs(lowres_inp).min())/(torch.abs(lowres_inp).max()-torch.abs(lowres_inp).min())
                
                writer.add_image("imagesoutput", torch.unsqueeze(output[0],0), iters) 
                writer.add_image("truthImage", torch.unsqueeze(truth_inp[0],0), iters)
                writer.add_image("lowresImage", torch.unsqueeze(lowres_inp[0],0), iters)
                
        iters += 1
        print("iters", iters) 
    torch.save({'epoch':epoch,
               'model_state_dict':net.state_dict(),
               'optimizer_state_dict': optimizer.state_dict(),
               'loss':loss,
              }, checkpoint_path+"/model_"+str(epoch)+".pt")


        
