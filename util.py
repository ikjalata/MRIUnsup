import os
import random
import numpy as np
import glob
import time

import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import transforms,utils,datasets
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from mri_util import cfl, recon
import util_torch
class MRI_Dataset(Dataset):
    def __init__(self, data_dir='', mask_dir='', data_type="train",transform=None):
        self.train_data = []
        self.transform = transform
        self.file_names = self.prepare_file_names(data_dir+"/"+data_type, str_search="/*.npy")
        self.mask_file_names = self.prepare_file_names(mask_dir, "/*.cfl")
        
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fl_path = self.file_names[idx]
        ksp_truth, ksp_input, sensemp, masks, mask_recon, scale= self.preprocess(fl_path)
        
        ksp_input = ksp_input.permute(2,0,1) 
        ksp_truth = ksp_truth.permute(2,0,1)
        sensemp = sensemp.permute(2,0,1) 
        
        lowres_img = self.transpose_model(ksp_input, sensemp);#np.save("foldrrr/lowres_img_yeak.npy", lowres_img)#;quit()
        #print("lowres_img", lowres_img.shape)
        lowres_img = self.complex_to_channels(lowres_img.squeeze())
        
        truth_img1 = self.transpose_model(ksp_truth, sensemp)
        truth_img = self.complex_to_channels(truth_img1.squeeze())
        ksp_truth = ksp_truth.permute(1,2,0)
        ksp_input = ksp_input.permute(1,2,0)
        sensemp = sensemp.permute(1,2,0)
         
        return ksp_truth, ksp_input, truth_img, lowres_img, sensemp, masks #, truth_img1

    def prepare_file_names(self,data_path, str_search = "/.*npy"):
        '''Find and return filenames of the given path'''
        if not os.path.exists(data_path):
            raise FileNotFoundError("the folder is not found:", data_path)
        #print("data_path", data_path+str_search)
        data_list = glob.glob(data_path + str_search)
        random.shuffle(data_list)

        return data_list
    def load_masks_cfl(self, filenames, image_shape=None):
        """Read masks from files."""
        if image_shape is None:
           # First find masks shape...
           image_shape = [0, 0]
           for f in filenames:
               f_cfl = os.path.splitext(f)[0]
               mask = np.squeeze(cfl.read(f_cfl))
               shape_z = mask.shape[-2]
               shape_y = mask.shape[-1]
               if image_shape[-2] < shape_z:
                  image_shape[-2] = shape_z
               if image_shape[-1] < shape_y:
                  image_shape[-1] = shape_y

        masks = np.zeros([len(filenames)] + image_shape, dtype=np.complex64)

        i_file = 0
        for f in filenames:
            f_cfl = os.path.splitext(f)[0]
            tmp = np.squeeze(cfl.read(f_cfl))
            tmp = recon.zeropad(tmp, image_shape)
            masks[i_file, :, :]= tmp
            i_file = i_file + 1
        
        return masks
    #load the data and masks and preprocess them
    def preprocess(self, file_path):
        #load mask with [36,256,320] shape and choose randomly one mask [256,320]
        masks = torch.tensor(self.load_masks_cfl(self.mask_file_names))
        mask_x2=masks
        masks = self.mask_process(masks)
        masks = masks.unsqueeze(0)
        # flip up down is not necessary since the size of the masks 1,kz,ky
        #flip left to right => tf.image.random_flip_left_right(mask_x, seed=random_seed)
        #print(masks.shape, )
        idx = torch.randperm(2)
        if idx[0]==0: 
           masks = torch.flip(masks, [1])
        else:
           masks = masks

        #load .npy data and change to tensor
        data = np.load(file_path, allow_pickle=True)
        kspace = data.item().get('ks')
        sensemap = data.item().get('map')
        sensemap = np.squeeze(sensemap) 
        
        kspace = torch.from_numpy(kspace)
        sensemap = torch.from_numpy(sensemap)
        
        kspace = kspace.permute(1,2,0)
        sensemap = sensemap.permute(1,2,0)
        masks = masks.permute(1,2,0) 
        #print("kspace, sensemap.shape", kspace.shape, sensemap.shape);quit() 
        
        shape_sc=5
        w=256
        z=320
        d1=(w-shape_sc)//2
        d2=w-shape_sc-d1
        d11=(z-shape_sc)//2
        d22=z-shape_sc-d11
        mask_calib = torch.ones(shape_sc, shape_sc,1,dtype=torch.complex64)#;print("d1,d2",d1,d2,d11,d22)
        #pad(left, right, top, bottom)
        mask_calib = F.pad(input=mask_calib, pad=(0, 0, d11, d22, d1, d2), mode='constant', value=0)
        #print("maskcalibDim", mask_calib.shape, masks.shape)
        #mask_calib = mask_calib.squeeze()
        mask_x = masks * (1 - mask_calib) + mask_calib
        
        mask_recon = torch.abs(kspace) / torch.max(torch.abs(kspace))
        mask_recon = (mask_recon>1e-7).type(torch.complex64)
        #mask_x = mask_x.unsqueeze(-1);print("hey",mask_x.shape, mask_recon.shape)
        mask_x = mask_x * mask_recon
        
        x=w//2
        y=z//2
        m=shape_sc//2
        n=shape_sc-m
        scale = kspace[x-n:x+m,y-n:y+m,:]
        scale = torch.mean(torch.square(torch.abs(scale)))*(shape_sc*shape_sc/1e5)
        scale = 1.0/torch.sqrt(scale)
        scale = scale.type(torch.complex64)
        kspace = kspace * scale
        ks_truth = kspace
        ks_input = torch.mul(kspace, mask_x)
        
        return ks_truth, ks_input, sensemap, mask_x, mask_recon, scale
    #change kspace to image using inverse fft
    def transpose_model2(self, kspace, sensemap):
        lowres_img = torch.fft.ifft2(kspace)
        lowres_img = torch.mul(torch.conj(sensemap), lowres_img)
        lowres_img = torch.sum(lowres_img, 0)
        lowres_img = torch.unsqueeze(lowres_img, dim=-1)
        return lowres_img

    def transpose_model(self, kspace, sensemap):
        index = torch.tensor(kspace.shape[-2:])
        scale = torch.sqrt(torch.prod(index));#print("scaleiii",scale) 
        axes = (-2, -1)
        tmp = torch.fft.fftshift(kspace, dim=axes)
        tmp = torch.fft.ifftn(tmp, dim=axes)
        tmp = torch.fft.ifftshift(tmp, dim=axes)*scale

        lowres_img = torch.mul(torch.conj(sensemap), tmp)
        lowres_img = torch.sum(lowres_img, 0)
        lowres_img = torch.unsqueeze(lowres_img, dim=-1)
        return lowres_img

    def model_forward2(self, image, sensemap):
        lowres_img = torch.mul(torch.conj(sensemap), image)
        lowres_img = torch.sum(lowres_img, 0)
        kspace = torch.fft.ifft2(lowres_img)

        return kspace

    def model_forward(self, low_image, sensemap):
        image = torch.mul(sensemap, low_image)
        image = torch.sum(image, 0)

        index = torch.tensor(low_image.shape[-2:])
        scale = torch.sqrt(torch.prod(index))
        axes = (-2, -1)
        tmp = torch.fft.fftshift(image, dim=axes)
        tmp = torch.fft.fftn(tmp, dim=axes)
        kspace = torch.fft.ifftshift(tmp, dim=axes)/scale

        return kspace

    def complex_to_channels(self, image):
        out_image = torch.stack((torch.real(image), torch.imag(image)), axis=-1)
        out_image = out_image.squeeze()

        return out_image
    def channels_to_complex(self, image):
        out_image = torch.reshape(image, [-1, 2])
        out_image = torch.complex(image_out1[:, 0], image_out1[:, 1])
        out_shape[-1] = out_shape[-1] // 2 
        out_image = torch.reshape(out_image, out_shape)

        return out_image

    def mask_process(self, mask):
        #print("shap",mask.shape, type(mask))
        #idx = torch.randperm(mask.shape[0])
        #mask = mask[idx].view(mask.shape)
        #mask = mask[0,:]  #choose one mask out of size=file_size
        idx = torch.randperm(mask.size()[0])
        mask=mask[idx]
        #mask_x = tf.image.random_flip_up_down(mask_x, seed=random_seed)
        #mask_x = tf.image.random_flip_left_right(mask_x, seed=random_seed)
        
        #mask = mask.permute(1,2,0)   #transpose
      
        return mask[0] 
    def sumofsq(self, image):
        image = torch.square(torch.abs(image))
        image = torch.sum(image, -1, keepdim=True)
        image = torch.sqrt(image)

        return image




 
