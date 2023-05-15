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

def save_img(intensity_multi, mask_list, model_name, index):
    mask = mask_list[index]
    scene = os.path.splitext(os.path.basename(img_files[index]))[0]
    plt.imsave(SAVE_PATH + f'{scene}_{model_name}.png', intensity_multi[index]*mask, vmax=VMAX_INTENSITY, cmap='gray')
        
def save_intensity_error(intensity_multi, renderer, mask_list, index):
    f = open(SAVE_PATH + f'intensity_error.txt', mode='w')

    mask = mask_list[index]
    scene = os.path.splitext(os.path.basename(img_files[index]))[0]
    observed_average = np.mean(intensity_multi[index, mask])

    errormap = np.abs(intensity_multi[index] - renderer.intensity_multi[index])

    print(f'{scene}')
    print(f'RMSE: {np.sqrt(np.mean(errormap[mask]**2))}')
    print(f'RMSE devided by the mean of observed intensity: {np.sqrt(np.mean(errormap[mask]**2))/observed_average}')
    
    f.write(f'{scene}\n')
    f.write(f'RMSE: {np.sqrt(np.mean(errormap[mask]**2))}\n')
    f.write(f'RMSE devided by the mean of observed intensity: {np.sqrt(np.mean(errormap[mask]**2))/observed_average}\n')
    f.close()

def get_option():
    argparser = argparse.ArgumentParser(description='Render with estimated BRDF')
    argparser.add_argument('OBJECT_NAME')
    argparser.add_argument('INDEX', type=int)
    return argparser.parse_args()
# command line arguments
args = get_option()
obj_name = args.OBJECT_NAME
index = args.INDEX

# settings
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
distribution_function = FMBRDF.generalized_gauss_distribution
net_body, net_shadowing = FMBRDF.load_net(device)
step_c = 100000
bright_threshold = 1
dark_threshold = 0.01
VMAX_INTENSITY = 0.5

# directory, file
DATA_PATH = '../data/'
IMAGE_PATH = DATA_PATH+obj_name+'/image/img_ud/'
MASK_PATH = DATA_PATH+obj_name+'/'
v_file = DATA_PATH+obj_name+'/calibration/vpL_img.npy'
light_file = DATA_PATH+obj_name+'/calibration/L.txt'
normal_GT_file = DATA_PATH+obj_name+'/normal/normal_GT.npy'
SAVE_PATH = DATA_PATH+obj_name+f'/estimation_BRDF/'
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
h, w = mask.shape

# load images
img_files = sorted(glob.glob(IMAGE_PATH+'*.png'))
intensity_multi = np.zeros((l_num,h,w))
mask_list_error = np.zeros((l_num,h,w), dtype=bool)
mask_list_rendering = np.zeros((l_num,h,w), dtype=bool)
for i, img_file in enumerate(img_files):
    scene = os.path.splitext(os.path.basename(img_file))[0]
    pimg = cv2.imread(img_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE).astype(np.float32) /(256*256-1.0)

    pbayer_ = ps.PBayer(pimg)
    I_DC = pbayer_.I_dc
    intensity = 2*I_DC
    intensity_multi[i] = intensity
    mask_list_error[i] = mask & (pimg[::2,::2] < bright_threshold) & (pimg[1::2,::2] < bright_threshold) & (pimg[::2,1::2] < bright_threshold) & (pimg[1::2,1::2] < bright_threshold) & (I_DC > dark_threshold)
    mask_list_rendering[i] = mask & (I_DC > dark_threshold)
save_img(intensity_multi, mask_list_rendering, 'observation', index)

# rendering
# load estimated parameters
model_name = 'FMBRDF'
LOAD_PATH = DATA_PATH+obj_name+f'/estimation_BRDF/'
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

save_img(renderer_BRDF.intensity_multi, mask_list_rendering, model_name, index)
save_intensity_error(intensity_multi, renderer_BRDF, mask_list_error, index)