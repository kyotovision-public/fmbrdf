import struct
import math
import sys
import os
import numpy as np
import numba
from numba import jit, double, float32, int32, boolean, typeof, prange
from numba.experimental import jitclass
from numba.types import Tuple, List

sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))
from utils.my_math import Rusinkiewicz_transform

def read_MERL_binary(merl_file):
    '''
    return reshaped BRDF ndarray shaped into (theta_h, theta_d, phi_d, color)
    '''
    sampling_theta_h = 90
    sampling_theta_d = 90
    sampling_phi_d = 180
    RED_SCALE = 1.0/1500
    GREEN_SCALE = 1.15/1500
    BLUE_SCALE = 1.66/1500

    with open(merl_file, 'rb') as f:
        data = f.read()
        length = sampling_theta_h * sampling_theta_d * sampling_phi_d
        n = struct.unpack_from('3i', data) # first 3 elements of binary file are the number of sampling of theta_h, theta_d, phi_d

        if  n[0]*n[1]*n[2] != length:
            raise IOError("Dimmensions does not match")

        brdf = struct.unpack_from(str(3*length)+'d', data,
                                        offset=struct.calcsize('3i'))
    brdf = np.array(brdf).reshape(3, sampling_theta_h, sampling_theta_d, sampling_phi_d).transpose(1,2,3,0)
    brdf[...,0] *= RED_SCALE
    brdf[...,1] *= GREEN_SCALE
    brdf[...,2] *= BLUE_SCALE

    return brdf

def _get_sampled_angles(sampling_theta_h=90, sampling_theta_d=90, sampling_phi_d=180):
    '''
    get sampled theta_h, theta_d, and phi_d in MERL BRDF dataset
    '''
    idx = np.arange(sampling_theta_h).astype(np.float64) / sampling_theta_h
    theta_h = idx * idx * np.pi/2.0

    idx = np.arange(sampling_theta_d).astype(np.float64) / sampling_theta_d
    theta_d = idx * np.pi/2.0

    idx = np.arange(sampling_phi_d).astype(np.float64) / sampling_phi_d
    phi_d = idx * np.pi

    return theta_h, theta_d, phi_d

@jit
def _get_valid_angles(brdf, theta_h, th_min, radius_th_max=0.9):
    '''
    remove the area where brdf does not have valid value or theta_h is too small or large
    brdf.shape = theta_h.shape + RGB
    '''
    assert brdf.shape[0:-1] == theta_h.shape

    # get rid of copletely shadowing area
    mask = (brdf[...,0] > 0) & (brdf[...,1] > 0) & (brdf[...,2] > 0)

    # get rid of where theta_h < th_min
    mask = mask & (theta_h > th_min)

    # get rid of where theta_h > th_max
    th_temp = theta_h[mask]
    mask_radius = np.sin(theta_h) < (radius_th_max*np.max(np.sin(th_temp)))
    mask = mask & mask_radius

    return brdf[mask], theta_h[mask]

@jit
def _load_BRDF_angles(brdf, theta_h, theta_d, phi_d, th_min, radius_th_max):
    # mask with theta_h
    len_th = theta_h.shape[0]
    len_td = theta_d.shape[0]
    len_pd = phi_d.shape[0]
    print(len_th, len_td, len_pd)
    ret_brdf = np.zeros((len_th, len_td, len_pd, 3))
    ret_theta_h = np.zeros((len_th, len_td, len_pd))
    ret_theta_d = np.zeros((len_th, len_td, len_pd))
    ret_phi_d = np.zeros((len_th, len_td, len_pd))
    ret_mask = np.zeros((len_th, len_td, len_pd))

    for i_td in range(len_td):
        for i_pd in range(len_pd):
            brdf_temp, theta_h_temp = _get_valid_angles(brdf[:, i_td, i_pd], theta_h, th_min, radius_th_max)

            size = len(theta_h_temp)
            ret_brdf[:size, i_td, i_pd] = brdf_temp
            ret_theta_h[:size, i_td, i_pd] = theta_h_temp
            ret_theta_d[:size, i_td, i_pd] = theta_d[i_td]
            ret_phi_d[:size, i_td, i_pd] = phi_d[i_pd]
            ret_mask[:size, i_td, i_pd] = 1.0
    return ret_brdf, ret_theta_h, ret_theta_d, ret_phi_d, ret_mask

def load_BRDF_angles(merl_file, th_min, td_max, radius_th_max=0.9, step_th=1, step_td=1, step_pd=1, sampling_theta_h=90, sampling_theta_d=90, sampling_phi_d=180):
    '''
    load BRDF and corresponding angles(theta_h, theta_d, phi_d) removing unreliable values
    '''

    # load BRDF
    brdf = read_MERL_binary(merl_file)
    brdf = brdf[::step_th, ::step_td, ::step_pd]

    theta_h, theta_d, phi_d = _get_sampled_angles(sampling_theta_h, sampling_theta_d, sampling_phi_d)
    theta_h = theta_h[::step_th]
    theta_d = theta_d[::step_td]
    phi_d = phi_d[::step_pd]
    
    # mask with theta_d
    mask_td = theta_d < td_max
    theta_d = theta_d[mask_td]
    brdf = brdf[:,mask_td,:,:]

    ret_brdf, ret_theta_h, ret_theta_d, ret_phi_d, ret_mask = _load_BRDF_angles(brdf, theta_h, theta_d, phi_d, th_min, radius_th_max)
    mask = ret_mask > 0.5
    return ret_brdf[mask], ret_theta_h[mask], ret_theta_d[mask],ret_phi_d[mask]

def load_BRDF_th(merl_file, i_td, i_pd, th_min, radius_th_max=0.9, sampling_theta_h=90, sampling_theta_d=90, sampling_phi_d=180):
    '''
    load BRDF and corresponding theta_h with given theta_d, phi_d (i_td, i_pd)
    '''

    # load BRDF
    brdf = read_MERL_binary(merl_file)

    theta_h, theta_d, phi_d = _get_sampled_angles(sampling_theta_h, sampling_theta_d, sampling_phi_d)

    # mask with theta_h
    ret_brdf, ret_theta_h = _get_valid_angles(brdf[:, i_td, i_pd], theta_h, th_min, radius_th_max)
    return ret_brdf, ret_theta_h, theta_d[i_td], phi_d[i_pd]
