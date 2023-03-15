import torch 
from mri_util import cfl, recon
import numpy as np 
import os
import glob
import random
import torch.nn.functional as F


def complex_to_channels(image):
    out_image = torch.stack((torch.real(image), torch.imag(image)), axis=-1)
    
    return out_image


def channels_to_complex(imag):
    imgshape = imag.shape
    out_image = torch.reshape(imag, [imgshape[0],-1, 2])
    out_image = torch.complex(out_image[:, :, 0], out_image[:, :, 1])
    out_shape = imag.shape 
    out_image = torch.reshape(out_image, [imgshape[0], imgshape[1], imgshape[2]])

    return out_image
    
    
def transpose_model2(kspace, sensemap):
    lowres_img = torch.fft.ifft2(kspace)
    lowres_img2 = torch.mul(torch.conj(sensemap), lowres_img)
    lowres_img2 = torch.sum(lowres_img2, 1)
    
    return lowres_img2


def transpose_model(kspace, sensemap):
    index = torch.tensor(kspace.shape[-2:])
    scale = torch.sqrt(torch.prod(index))
    axes = (-2, -1)
    tmp = torch.fft.fftshift(kspace, dim=axes)
    tmp = torch.fft.ifftn(tmp, dim=axes)
    tmp = torch.fft.ifftshift(tmp, dim=axes)*scale

    lowres_img2 = torch.mul(torch.conj(sensemap), tmp)
    lowres_img2 = torch.sum(lowres_img2, 1)

    return lowres_img2


def model_forward2(image, sensemap):
    lowres_img = torch.mul(torch.conj(sensemap), image)
    lowres_img = torch.sum(lowres_img, 1)
    kspace = torch.fft.ifft2(lowres_img)

    return kspace

def model_forward(low_image, sensemap):
    low_image = torch.unsqueeze(low_image, 1) 
    image = torch.mul(sensemap, low_image)
    
    index = torch.tensor(low_image.shape[-2:])
    scale = torch.sqrt(torch.prod(index))
    axes = (-2, -1)
    tmp = torch.fft.fftshift(image, dim=axes)
    tmp = torch.fft.fftn(tmp, dim=axes)/scale 
    kspace = torch.fft.fftshift(tmp, dim=axes) 
    
    return kspace


def sumofsq(image):
    image = torch.square(torch.abs(image))
    image = torch.sum(image, -1, keepdim=True)
    image = torch.sqrt(image)

    return image_out


def load_masks_cfl(filenames, image_shape=None):
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


def mask_process(mask):
    idx = torch.randperm(mask.size()[0])
    mask=mask[idx]
  
    return mask[0]
    
    
def prepare_file_names(data_path, str_search = "/.*npy"):
    '''Find and return filenames of the given path'''
    if not os.path.exists(data_path):
        raise FileNotFoundError("the folder is not found:", data_path)
    data_list = glob.glob(data_path + str_search)
    random.shuffle(data_list)

    return data_list
    
    
def measure(X_gen, sensemap, mask_dir, device):
    #load mask with [36,256,320] shape and choose randomly one mask [256,320]
    mask_file_names = prepare_file_names(mask_dir, "/*.cfl")
    masks_orig = torch.tensor(load_masks_cfl(mask_file_names))
    
    image = channels_to_complex(X_gen.permute(0,2,3,1))
    kspace_t = model_forward(image, sensemap)
    for i in range(image.size(0)):
        kspace = kspace_t[i]
        mask_x2=masks_orig
        masks = mask_process(masks_orig)
        masks = masks.unsqueeze(0)
        
        idx = torch.randperm(2)
        if idx[0]==0: 
            masks = torch.flip(masks, [1])
        else:
            masks = masks

        
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
        #mask_calib = mask_calib.squeeze()
        mask_x = masks * (1 - mask_calib) + mask_calib
        
        mask_recon = torch.abs(kspace) / torch.max(torch.abs(kspace))
        mask_recon = (mask_recon>1e-7).type(torch.complex64)
        
        mask_x = mask_x.permute(2,0,1).to(device) * mask_recon
        
        x=w//2
        y=z//2
        m=shape_sc//2
        n=shape_sc-m
        scale = kspace[x-n:x+m,y-n:y+m,:];#print(kspace.shape,scale.detach().cpu().numpy());quit()
        scale = torch.mean(torch.square(torch.abs(scale)))*(shape_sc*shape_sc/1e5)
        scale = 1.0/torch.sqrt(scale)
        scale = scale.type(torch.complex64)
        #print("scale:::::", scale)#,abs(kspace.numpy()).max()) 
        kspace = kspace #* scale
        ks_truth = kspace;
        ks_input = torch.mul(kspace, mask_x)
        if (i==0):
           total_kspace = ks_input
        else:
            total_kspace = torch.stack([total_kspace, ks_input])
        x_measured = transpose_model(total_kspace, sensemap)
        x_measured = complex_to_channels(x_measured)
        
    
    return x_measured.permute(0,3,1,2)


