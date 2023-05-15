import numpy as np
import cv2
import os
import sys
import glob
import argparse
import torch
import torch.nn as nn

sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))
import BRDFmodel.FMBRDF as FMBRDF
import utils.pimg_smooth_mono as ps

def get_option():
    argparser = argparse.ArgumentParser(description='fitting FMBRDF with intensity images')
    argparser.add_argument('OBJECT_NAME')
    return argparser.parse_args()
# command line arguments
args = get_option()
obj_name = args.OBJECT_NAME

# settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
distribution_function = FMBRDF.generalized_gauss_distribution
net_body, net_shadowing = FMBRDF.load_net(device)
step_c = 100000
lr = 0.003
iteration=10000
bright_threshold = 1
dark_threshold = 0.01
loss_fn = FMBRDF.mse_loss_intensity()


# initial values
refractive_index_init = 1.5
albedo_ratio_init = 1.0
params_init = np.array([np.pi/12, 2.0])
k_s_init = 1.0
kappa_init = 0

# directory, file
DATA_PATH = '../data/'
IMAGE_PATH = DATA_PATH+obj_name+'/image/img_ud/'
MASK_PATH = DATA_PATH+obj_name+'/'
v_file = DATA_PATH+obj_name+'/calibration/vpL_img.npy'
light_file = DATA_PATH+obj_name+'/calibration/L.txt'
normal_GT_file = DATA_PATH+obj_name+'/normal/normal_GT.npy'
SAVE_PATH = DATA_PATH+obj_name+f'/fitting-intensity/'
save_file = SAVE_PATH + 'params.npz'
os.makedirs(SAVE_PATH, exist_ok=True)

# load files
v = -np.load(v_file)
l_list = np.loadtxt(light_file, delimiter=',')
l_num = len(l_list)
n_GT = np.load(normal_GT_file)
# load mask
mask_full = np.load(MASK_PATH + 'mask_ud.npy')
mask = mask_full[::2,::2]
kernel = np.ones((5,5),np.uint8)
mask = cv2.erode(mask, kernel, iterations = 1) >= 1
mask_list = np.tile(mask,(l_num, 1,1))
h, w = mask.shape

# load images
img_files = sorted(glob.glob(IMAGE_PATH+'*.png'))
observed_stok_multi = np.zeros((l_num, h, w, 3))
for i, img_file in enumerate(img_files):
    print(l_list[i])
    pimg = cv2.imread(img_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE).astype(np.float32) /(256*256-1.0)

    pbayer_ = ps.PBayer(pimg)
    I_DC = pbayer_.I_dc
    DoLP = pbayer_.DoLP
    AoLP = -pbayer_.AoLP
    observed_stok_multi[i] = np.stack([2*I_DC, 2*I_DC*DoLP*np.cos(2*AoLP), 2*I_DC*DoLP*np.sin(2*AoLP)], axis=2)
    mask_list[i] = mask & (pimg[::2,::2] < bright_threshold) & (pimg[1::2,::2] < bright_threshold) & (pimg[::2,1::2] < bright_threshold) & (pimg[1::2,1::2] < bright_threshold) & (I_DC > dark_threshold)

refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est = FMBRDF.BRDFestimation_noWeight(distribution_function, observed_stok_multi, n_GT, l_list, v, refractive_index_init, albedo_ratio_init, k_s_init, params_init, kappa_init, mask_list, step_c, net_body, net_shadowing, loss_fn, opt_ks=True, lr=lr, iteration=iteration)

print('refractive_index_est: ', refractive_index_est)
print('albedo_ratio_est: ', albedo_ratio_est)
print('k_s_est: ', k_s_est)
print('params_est: ', params_est)
print('kappa_est: ', kappa_est)

np.savez_compressed(save_file, n=n_GT, l_list=l_list, v=v, refractive_index_est=refractive_index_est, albedo_ratio_est=albedo_ratio_est, k_s_est=k_s_est, params_est=params_est, kappa_est=kappa_est, mask_list=mask_list)