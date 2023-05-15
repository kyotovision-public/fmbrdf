import numpy as np
import cv2
import os
import sys
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))
import BRDFmodel.FMBRDF as FMBRDF

def get_option():
    argparser = argparse.ArgumentParser(description='Preprocess')
    argparser.add_argument('OBJECT_NAME')
    return argparser.parse_args()
args = get_option()
obj_name = args.OBJECT_NAME

# settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
distribution_function = FMBRDF.generalized_gauss_distribution
net_body, net_shadowing = FMBRDF.load_net(device)
v = np.array([0,0,-1.0])
step_c = 100000
lr = 0.01
iteration=100000
fontsize = 18

# initial values
refractive_index_init = 1.5
albedo_ratio_init = 1.0
k_s_init = 1.0
params_init = np.array([np.pi/12, 2.0])
kappa_init = 20

# directory, file
DATA_PATH = '../data/'
data_file = DATA_PATH + obj_name + '/data.npz'
SPHERE_PATH = DATA_PATH + obj_name + '_sphere/'

# load intensity, camera pose, normal
data_npz = np.load(data_file)
valid_mask = data_npz['valid_mask']
stokes_list = data_npz['stokes_list'][valid_mask]
intensity_list = data_npz['intensity_list'][valid_mask]
n_list = data_npz['normal_list'][valid_mask]
l_list = data_npz['light_list'][valid_mask]
l_num = np.sum(valid_mask)

# filename
model_name = 'FMBRDF'
save_file = DATA_PATH + obj_name + f'/{model_name}_params.npz'
save_img = DATA_PATH + obj_name + f'/{model_name}_intensity.pdf'

# fitting
refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est = FMBRDF.BRDFfitting_plane(distribution_function, stokes_list, n_list, l_list, v, refractive_index_init, albedo_ratio_init, k_s_init, params_init, kappa_init, step_c, net_body, net_shadowing, lr=lr, iteration=iteration, epsilon=1e-6, isPrint=True)
print('refractive_index_est: ', refractive_index_est)
print('albedo_ratio_est: ', albedo_ratio_est)
print('k_s_est: ', k_s_est)
print('params_est: ', params_est)
print('kappa_est: ', kappa_est)
np.savez_compressed(save_file, refractive_index_est=refractive_index_est, albedo_ratio_est=albedo_ratio_est, k_s_est=k_s_est, params_est=params_est, kappa_est=kappa_est)

# rendering
renderer_BRDF = FMBRDF.Renderer_plane(distribution_function, n_list, l_list, v, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est)
renderer_BRDF.rendering(net_body, net_shadowing, step_c)
theta_nv = np.arccos(np.clip(np.sum(n_list*v, axis=-1), -1, 1))*np.sign(-n_list[...,0])
plt.figure()
plt.plot(theta_nv*180/np.pi, renderer_BRDF.intensity, label='Ours', c='r')
plt.scatter(theta_nv*180/np.pi, intensity_list, label='Observation')
plt.xlabel(r'$\theta_{nv}$')
plt.ylabel('Intensity')
plt.legend()
plt.tick_params(labelsize=fontsize)
plt.savefig(save_img, bbox_inches="tight")
plt.close()
