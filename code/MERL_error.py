import sys
import os
import math
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import glob
import pickle

sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))
from utils.merl import load_BRDF_angles
import BRDFmodel.FMBRDF as FMBRDF
from utils.my_math import Rusinkiewicz2nl

def compute_log_RMSE(observed_BRDF, model_brdf):
    LogRMSE = np.sqrt(np.mean(np.square(np.log10(model_brdf + 1e-18) - np.log10(observed_BRDF + 1e-18))))
    return LogRMSE

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # save file
    PATH = f'MERL_fitting_result/'
    savefile = PATH + 'LogRMSE.pickle'

    # other parameters
    distribution_function = FMBRDF.generalized_gauss_distribution
    net_body, net_shadowing = FMBRDF.load_net(device)
    step_c = 100000
    # theta_h, theta_d, phi_d
    th_min = np.arccos(0.9999)
    radius_th_max = 1.0
    td_max = np.pi/2 + 1e-7
    step_td = 1
    step_pd = 1

    dict_LogRMSE = {}
    filenames = sorted(glob.glob('../MERL_BRDFDatabase/brdfs/*.binary'))
    for filename in filenames:
        obj_name = os.path.splitext(os.path.basename(filename))[0]
        print(obj_name)

        # save and load files
        MERLPATH = '../MERL_BRDFDatabase/brdfs/'
        MERLfilename = MERLPATH + obj_name + '.binary'
        npz_filename = PATH + f'{obj_name}_parameters.npz'

        # obtain observed BRDF, n, l, v
        brdf, theta_h, theta_d, phi_d = load_BRDF_angles(MERLfilename, th_min, td_max, radius_th_max, step_td = step_td, step_pd=step_pd)
        observed_BRDF = brdf
        v = np.array([0,0,-1.0])
        n, l = Rusinkiewicz2nl(theta_h, theta_d, phi_d, v)
        # masking the region that is not visible or lit
        mask_visible = (np.sum(n*v, axis=-1) > 1e-9) & (np.sum(n*l, axis=-1) > 1e-9)
        observed_BRDF = observed_BRDF[mask_visible]
        theta_h = theta_h[mask_visible]
        n = n[mask_visible]
        l = l[mask_visible]

        # load npz file
        npz_comp = np.load(npz_filename)
        refractive_index = npz_comp['refractive_index']
        albedo_ratio = npz_comp['albedo_ratio']
        k_s = npz_comp['k_s']
        dist_params = npz_comp['dist_params']
        kappa = npz_comp['kappa']
        # rendering
        renderer = FMBRDF.Renderer_BRDF(distribution_function, n, l, v, refractive_index, albedo_ratio, k_s, dist_params, kappa)
        renderer.rendering(net_body, net_shadowing, step_c)
        # compute error
        LogRMSE = compute_log_RMSE(observed_BRDF, renderer.model_brdf)
        print('LogRMSE', LogRMSE)

        dict_LogRMSE[obj_name] = LogRMSE

    # save result
    with open(savefile, "wb") as f:
        pickle.dump(dict_LogRMSE, f)

if __name__ == '__main__':
    main()