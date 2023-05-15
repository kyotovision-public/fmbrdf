import numpy as np
import cv2
import os
import sys
import glob
import argparse
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))
import BRDFmodel.FMBRDF as FMBRDF
import utils.pimg_smooth_mono as ps

def save_DoLP(DoLP_multi, mask_list, model_name):
    l_num = len(DoLP_multi)
    for i in range(l_num):
        mask = mask_list[i]
        scene = os.path.splitext(os.path.basename(img_files[i]))[0]
        plt.imsave(SAVE_PATH + f'DoLP/{scene}_{model_name}.png', DoLP_multi[i]*mask, vmax=VMAX_DOLP)

def save_DoLP_error(DoLP_multi, renderer, mask_list):
    f = open(SAVE_PATH + f'DoLP_error.txt', mode='w')

    l_num = len(DoLP_multi)
    errormap_list = np.zeros((l_num, ) + mask_list.shape[1:])
    for i in range(l_num):
        mask = mask_list[i]
        scene = os.path.splitext(os.path.basename(img_files[i]))[0]

        errormap = np.abs(DoLP_multi[i] - renderer.DoLP_multi[i])
        errormap_list[i] = errormap.copy()
        f.write(f'{scene}\n')
        f.write(f'RMSE: {np.sqrt(np.mean(errormap[mask]**2))}\n\n')

    print(f'RMSE: {np.sqrt(np.mean(errormap_list[mask_list]**2))}')
    f.write('Whole')
    f.write(f'RMSE: {np.sqrt(np.mean(errormap_list[mask_list]**2))}\n')
    f.close()

def save_DoLP_plot(DoLP_multi, mask_list, model_name, n, v, ylims, xlim=[0, 90]):
    l_num = len(DoLP_multi)
    for i in range(l_num):
        mask = mask_list[i]
        scene = os.path.splitext(os.path.basename(img_files[i]))[0]
        ylim = ylims[i]
        
        plt.figure(dpi=100)
        plot_x = (180/np.pi)*np.arccos(np.clip(np.sum(n*v, axis=-1), 0.0, 1.0))
        plot_x = plot_x[mask]
        plot_y = DoLP_multi[i, mask]
        plt.scatter(plot_x, plot_y, s=1, c=color_dict[model_name])
        plt.xlabel(r'$cos^{-1} (N \cdot V)$')
        plt.ylabel('DoLP')
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.savefig(SAVE_PATH + f'DoLP_plot/{scene}_nv_{model_name}.png', bbox_inches="tight")
        plt.close()

def get_option():
    argparser = argparse.ArgumentParser(description='Render with fitted BRDF')
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
bright_threshold = 1.1
dark_threshold = 0.01
VMAX_DOLP = 0.5
color_dict = {'observation':'c', 'FMBRDF':'r'}

# directory, file
DATA_PATH = '../data/'
IMAGE_PATH = DATA_PATH+obj_name+'/image/img_ud/'
MASK_PATH = DATA_PATH+obj_name+'/'
v_file = DATA_PATH+obj_name+'/calibration/vpL_img.npy'
light_file = DATA_PATH+obj_name+'/calibration/L.txt'
normal_GT_file = DATA_PATH+obj_name+'/normal/normal_GT.npy'
SAVE_PATH = DATA_PATH+obj_name+f'/fitting-DoLP/'
save_file = SAVE_PATH + 'params.npz'
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(SAVE_PATH+'DoLP/', exist_ok=True)
os.makedirs(SAVE_PATH+'DoLP_plot/', exist_ok=True)

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
h, w = mask.shape

# load images
img_files = sorted(glob.glob(IMAGE_PATH+'*.png'))
DoLP_multi = np.zeros((l_num,h,w))
mask_list_error = np.zeros((l_num,h,w), dtype=bool)
mask_list_rendering = np.zeros((l_num,h,w), dtype=bool)
for i, img_file in enumerate(img_files):
    scene = os.path.splitext(os.path.basename(img_file))[0]
    pimg = cv2.imread(img_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE).astype(np.float32) /(256*256-1.0)

    pbayer_ = ps.PBayer(pimg)
    I_DC = pbayer_.I_dc
    DoLP = pbayer_.DoLP
    DoLP_multi[i] = DoLP
    mask_list_error[i] = mask & (pimg[::2,::2] < bright_threshold) & (pimg[1::2,::2] < bright_threshold) & (pimg[::2,1::2] < bright_threshold) & (pimg[1::2,1::2] < bright_threshold) & (I_DC > dark_threshold)
    mask_list_rendering[i] = mask & (I_DC > dark_threshold)

# range of plot
ylims = np.zeros((l_num, 2))
for i in range(l_num):
    mask = mask_list_rendering[i]
    ylims[i,1] = np.quantile(DoLP_multi[i, mask], 0.999) + 0.1

save_DoLP(DoLP_multi, mask_list_rendering, 'observation')
save_DoLP_plot(DoLP_multi, mask_list_rendering, 'observation', n_GT, v, ylims=ylims)

# load estimated parameters
model_name = 'FMBRDF'
LOAD_PATH = DATA_PATH+obj_name+f'/fitting-DoLP/'
load_file = LOAD_PATH + 'params.npz'
npz_comp = np.load(load_file)
refractive_index_est = npz_comp['refractive_index_est']
albedo_ratio_est = npz_comp['albedo_ratio_est']
k_s_est = npz_comp['k_s_est']
params_est = npz_comp['params_est']
kappa_est = npz_comp['kappa_est']

# rendering
renderer_BRDF = FMBRDF.Renderer(distribution_function, n_GT, l_list, v, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est, mask_list_rendering)
renderer_BRDF.rendering(net_body, net_shadowing, step_c)

save_DoLP(renderer_BRDF.DoLP_multi, mask_list_rendering, model_name)
save_DoLP_error(DoLP_multi, renderer_BRDF, mask_list_error)
save_DoLP_plot(renderer_BRDF.DoLP_multi, mask_list_rendering, model_name, n_GT, v, ylims=ylims)
