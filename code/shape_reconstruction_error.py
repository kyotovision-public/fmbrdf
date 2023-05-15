import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
import argparse
sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))
from utils.my_math import getNormalImg

def get_option():
    argparser = argparse.ArgumentParser(description='Compute error of shape reconstruction')
    argparser.add_argument('OBJECT_NAME')
    return argparser.parse_args()

args = get_option()
obj_name = args.OBJECT_NAME

model_names = ['Lambertian', 'FMBRDF']
vmax_error = 30


DATA_PATH = '../data/'
MASK_PATH = DATA_PATH+obj_name+'/'
SAVE_PATH = DATA_PATH+obj_name+'/shape_reconstruction/'
v_file = DATA_PATH+obj_name+'/calibration/vpL_img.npy'
normal_GT_file = DATA_PATH+obj_name+'/normal/normal_GT.npy'

# load files
n_GT = np.load(normal_GT_file)
# load mask
mask_full = np.load(MASK_PATH + 'mask_ud.npy')
mask = mask_full[::2,::2]
kernel = np.ones((5,5),np.uint8)
mask = cv2.erode(mask, kernel, iterations = 1) >= 1

remove_mask = (np.linalg.norm(n_GT,axis=-1) < 0.1)  & (np.isnan(n_GT)[...,0])
mask = mask & (~remove_mask)

# save GT normal
plt.imsave(SAVE_PATH+f'normal_GT.png', getNormalImg(n_GT)*mask[...,np.newaxis])



# compute error
def comp_error(n_GT, n_est, mask):
    cos = np.clip(np.sum(n_GT*n_est, axis=-1), -1, 1)
    theta = np.arccos(cos)*180/np.pi
    mean = np.nanmean(theta[mask])
    median = np.nanmedian(theta[mask])
    std = np.nanstd(theta[mask])
    return theta, mean, median, std

# compute and save error
f = open(SAVE_PATH + 'error.txt', mode='w')
for model_name in model_names:
    print(model_name)
    load_file = SAVE_PATH + f'{model_name}_estimation.npz'
    npz_comp = np.load(load_file)
    n_est = npz_comp['n_est']
    remove_mask = (np.linalg.norm(n_est,axis=-1) < 0.1)  & (np.isnan(n_est)[...,0])
    mask = mask & (~remove_mask)
    
    error_map, mean, median, std = comp_error(n_GT, n_est, mask)
    error_map = np.nan_to_num(error_map, 0)
    print('mean: ', mean)
    print('median: ', median)
    
    f.write(f'{model_name}\n')
    f.write(f'mean: {mean}\n')
    f.write(f'median: {median}\n')
    f.write(f'std: {std}\n')
    f.write('\n')
    plt.imsave(SAVE_PATH+f'normal_{model_name}.png', getNormalImg(n_est)*mask[...,np.newaxis])
    plt.imsave(SAVE_PATH+f'normal_error{model_name}.png', error_map*mask, vmin=0, vmax=vmax_error)

f.close()