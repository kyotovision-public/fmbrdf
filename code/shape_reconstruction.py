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
import utils.photometric_stereo as photometric_stereo
import utils.pimg_smooth_mono as ps

def get_option():
    argparser = argparse.ArgumentParser(description='Shape reconstruction from multiple polarimetric images')
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
# criterion of convergence
epsilon=1e-4
epsilon_alter = 1e-3
epsilon_propagate = 1e-4
# parameters of loss functions
delta_huber_intensity = 0.01
delta_huber_DoLP = 0.01
delta_huber_polarization = 0.01
delta_huber_stok = delta_huber_intensity
loss_smooth_list = [FMBRDF.smoothness_loss(5, 0.05)]*3 + [FMBRDF.smoothness_loss(3, 0.03)]*3 + [FMBRDF.smoothness_loss(1, 0.01)]
loss_smooth_joint = FMBRDF.smoothness_loss(1, 0.0)
loss_smooth_joint_list = [FMBRDF.smoothness_loss(1, 0.0)]

# initial values
refractive_index_init = 1.5
albedo_ratio_init = 1.0
k_s_init = 1.0
params_init = np.array([np.pi/12, 2.0])
kappa_init = 0

# directory, file
DATA_PATH = '../data/'
IMAGE_PATH = DATA_PATH+obj_name+'/image/img_ud/'
MASK_PATH = DATA_PATH+obj_name+'/'
v_file = DATA_PATH+obj_name+'/calibration/vpL_img.npy'
light_file = DATA_PATH+obj_name+'/calibration/L.txt'
SAVE_PATH = DATA_PATH+obj_name+'/shape_reconstruction/'
os.makedirs(SAVE_PATH, exist_ok=True)

# load files
v = -np.load(v_file)
l_list = np.loadtxt(light_file, delimiter=',')
l_num = len(l_list)
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
# correction with light source intensities
if os.path.isfile(DATA_PATH+obj_name+'/calibration/intensities.npy'):
    intensities = np.load(DATA_PATH+obj_name+'/calibration/intensities.npy')
    intensities = intensities/np.max(intensities)
    observed_stok_multi = observed_stok_multi/intensities[...,None,None,None]
observed_intensity_multi = observed_stok_multi[...,0]



# photometric stereo
model_name = 'Lambertian'
save_file = SAVE_PATH + f'{model_name}_estimation.npz'
n_init = photometric_stereo.PS(observed_intensity_multi, mask_list, l_list)
np.savez_compressed(save_file, n_est=n_init, l_list=l_list, v=v, mask_list=mask_list)
# backfacing normal
n_init[n_init[...,2] > 0] = np.array([0,0,-1.0])
n_init[~mask] = 0.0



# shape reconstruction
model_name = 'FMBRDF'
save_file = SAVE_PATH + f'{model_name}_estimation.npz'
lam = 5.0
lam1 = 0.1
lam2 = 5.0
# BRDF is estimated from intensity and polarization
loss_fn_BRDF = FMBRDF.mse_loss_intensity_DoLP(lam1, loss1=nn.HuberLoss(delta=delta_huber_intensity), loss2=nn.HuberLoss(delta=delta_huber_DoLP))
loss_fn_PS = FMBRDF.mse_loss_stok(lam, loss1=nn.HuberLoss(delta=delta_huber_intensity), loss2=nn.HuberLoss(delta=delta_huber_stok))
loss_fn_joint = FMBRDF.mse_loss_intensity_DoLP_stok(lam1, lam2, loss1=nn.HuberLoss(delta=delta_huber_intensity), loss2=nn.HuberLoss(delta=delta_huber_DoLP), loss3=nn.HuberLoss(delta=delta_huber_stok))
loss_fn_alter = FMBRDF.mse_loss_intensity_DoLP_stok(lam1, lam2, loss1=nn.HuberLoss(delta=delta_huber_intensity), loss2=nn.HuberLoss(delta=delta_huber_DoLP), loss3=nn.HuberLoss(delta=delta_huber_stok))
loss_fn_pixel = FMBRDF.mse_loss_intensity_DoLP_stok_pixel(lam1, lam2, loss1=nn.HuberLoss(reduction='none', delta=delta_huber_intensity), loss2=nn.HuberLoss(reduction='none', delta=delta_huber_DoLP), loss3=nn.HuberLoss(reduction='none', delta=delta_huber_stok))

n_est, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est = FMBRDF.alternating_optimization_propagate(distribution_function, observed_stok_multi, n_init, l_list, v, refractive_index_init, albedo_ratio_init, k_s_init, params_init, kappa_init, mask_list, step_c, net_body, net_shadowing, loss_fn_BRDF=loss_fn_BRDF, loss_fn_PS=loss_fn_PS, loss_fn_joint=loss_fn_joint, loss_fn_alter=loss_fn_alter, loss_fn_pixel=loss_fn_pixel, opt_ks=True, lr=lr, iteration_each=iteration, iteration_joint=iteration, epsilon=epsilon, epsilon_alter=epsilon_alter, epsilon_propagate=epsilon_propagate, loss_smooth_list=loss_smooth_list, loss_smooth_joint=loss_smooth_joint, loss_smooth_joint_list=loss_smooth_joint_list)

print('refractive_index_est: ', refractive_index_est)
print('albedo_ratio_est: ', albedo_ratio_est)
print('k_s_est: ', k_s_est)
print('params_est: ', params_est)
print('kappa_est: ', kappa_est)

np.savez_compressed(save_file, n_est=n_est, l_list=l_list, v=v, refractive_index_est=refractive_index_est, albedo_ratio_est=albedo_ratio_est, k_s_est=k_s_est, params_est=params_est, kappa_est=kappa_est, mask_list=mask_list)
