import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import glob
import pickle
import itertools
import datetime

sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))
from utils.merl import load_BRDF_angles
import BRDFmodel.FMBRDF as FMBRDF
from utils.my_math import Rusinkiewicz2nl

def LogMSELoss_torch(pred, target):
    loss = torch.sqrt(torch.mean(torch.square(torch.log10(pred + 1e-18) - torch.log10(target + 1e-18))))
    return loss

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # sampling for initial estimation
    refractive_index_sampled = np.array([1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9])
    rho_sampled = np.logspace(-4, 0, 20)
    alpha_sampled = np.logspace(-2, np.log10(np.pi), 10)
    beta_sampled = np.logspace(np.log10(0.5), np.log10(4), 5)
    dist_params_sampled = np.array(np.meshgrid(alpha_sampled, beta_sampled)).T.reshape(-1, 2)
    k_s_sampled = np.logspace(-2, 0, 10)
    kappa_init = 0.0

    # other parameters
    distribution_function = FMBRDF.generalized_gauss_distribution
    net_body, net_shadowing = FMBRDF.load_net(device)
    step_c = 100000
    lr = 0.005
    epsilon = 1e-5
    loss_init = LogMSELoss_torch
    loss_fn = LogMSELoss_torch
    # theta_h, theta_d, phi_d
    th_min = np.arccos(0.9999)
    td_max = np.pi/2 + 1e-7
    radius_th_max = 1.0
    step_td = 1
    step_pd = 1

    # BRDF file
    filenames = sorted(glob.glob('../MERL_BRDFDatabase/brdfs/*.binary'))
    for filename in filenames:
        obj_name = os.path.splitext(os.path.basename(filename))[0]
        print(obj_name)

        # save and load files
        MERLPATH = '../MERL_BRDFDatabase/brdfs/'
        MERLfilename = MERLPATH + obj_name + '.binary'
        SAVEPATH = f'MERL_fitting_result/'
        savefile = SAVEPATH + f'{obj_name}_parameters'
        os.makedirs(SAVEPATH, exist_ok=True)

        # obtain BRDF, n, l, v
        brdf, theta_h, theta_d, phi_d = load_BRDF_angles(MERLfilename, th_min, td_max, radius_th_max, step_td = step_td, step_pd=step_pd)
        observed_BRDF = brdf
        v = np.array([0,0,-1.0])
        n, l = Rusinkiewicz2nl(theta_h, theta_d, phi_d, v)
        # masking the region that is not visible or lit.
        mask_visible = (np.sum(n*v, axis=-1) > 1e-9) & (np.sum(n*l, axis=-1) > 1e-9)
        observed_BRDF = observed_BRDF[mask_visible]
        theta_h = theta_h[mask_visible]
        n = n[mask_visible]
        l = l[mask_visible]
        print('num_brdf', len(theta_h))

        # optimization
        refractive_index_est, albedo_ratio_est, k_s_est, dist_params_est, kappa_est = FMBRDF.MERL_robust_optimization(distribution_function, observed_BRDF, n, l, v, refractive_index_sampled, rho_sampled, k_s_sampled, dist_params_sampled, kappa_init, step_c, net_body, net_shadowing, lr=lr, epsilon=epsilon, loss_fn=loss_fn, loss_init=loss_init)
        # save paramters
        np.savez_compressed(savefile, refractive_index=refractive_index_est, albedo_ratio=albedo_ratio_est, k_s=k_s_est, dist_params=dist_params_est, kappa=kappa_est)

        # output estimates
        print('refractive_index_est', refractive_index_est)
        print('albedo_ratio_est', albedo_ratio_est)
        print('k_s_est', k_s_est)
        print('dist_params_est', dist_params_est)
        print('kappa_est', kappa_est)

if __name__ == '__main__':
    main()
