import numpy as np
import cv2
import scipy as sc
import scipy.stats

import numba
from numba import jit, double, float32, int32, boolean, typeof, prange
from numba.experimental import jitclass
from numba.types import Tuple, List


def getSphereNormal(mask_sphe):
    h,w = mask_sphe.shape
    pixelPositions_all = np.mgrid[0:w, 0:h].T # h, w, 2

    # Find the center and radius of the sphere from the mask
    contours, hier = cv2.findContours(mask_sphe, 1, 2)
    cnt = contours[0]
    (cx, cy), cr = cv2.minEnclosingCircle(cnt)

    hx = pixelPositions_all[...,0]
    hy = pixelPositions_all[...,1]
    hr = np.sqrt( (hx-cx)**2 + (hy-cy)**2)

    sin_theta = hr/cr
    sin_theta[sin_theta>1]=1
    cos_theta = np.sqrt(1- sin_theta**2)

    # Compute the surface normal at the point
    n = np.zeros((h,w,3))
    n[...,0] = (hx-cx)/cr
    n[...,1] = (hy-cy)/cr
    n[...,2] = -cos_theta
    n = n/(np.linalg.norm(n, axis=-1, keepdims=True) + 1e-9)

    return n

def makeSphereNormal(img_shape, center,radius):
    img_shape = tuple(img_shape)
    
    mask_sphe = np.zeros(img_shape+(3,), dtype=np.uint8)
    cv2.circle(mask_sphe, center, radius, (255,255,255), thickness=-1)
    mask_sphe = cv2.cvtColor(mask_sphe, cv2.COLOR_BGR2GRAY)
    n = getSphereNormal(mask_sphe)
    mask = mask_sphe >= 1
    return mask, n

def getNormalImg(n):
    normal_img = n.copy()
    normal_img[:,:,1:3] = -normal_img[:,:,1:3]  
    normal_img= (normal_img+1)/2
    return normal_img


def get_rotation_around_xyz(rot, axis):
    assert (axis == 0) or (axis == 1) or (axis == 2)

    rot = np.array(rot)
    if rot.ndim == 0:
        R = np.zeros((3,3))
    else:
        R = np.zeros(rot.shape + (3,3))
    
    R[...,axis,axis] = 1.0
    R[...,(axis+1)%3, (axis+1)%3] = np.cos(rot)
    R[...,(axis+1)%3, (axis+2)%3] = -np.sin(rot)
    R[...,(axis+2)%3, (axis+1)%3] = np.sin(rot)
    R[...,(axis+2)%3, (axis+2)%3] = np.cos(rot)
    return R

@jit
def get_rotation_around_xyz_jit_pixel(rot, axis):
    assert (axis == 0) or (axis == 1) or (axis == 2)

    R = np.zeros((3,3))
    
    R[axis,axis] = 1.0
    R[(axis+1)%3, (axis+1)%3] = np.cos(rot)
    R[(axis+1)%3, (axis+2)%3] = -np.sin(rot)
    R[(axis+2)%3, (axis+1)%3] = np.sin(rot)
    R[(axis+2)%3, (axis+2)%3] = np.cos(rot)
    return R

@jit
def get_rotation_around_xyz_jit(rot, axis):
    assert (axis == 0) or (axis == 1) or (axis == 2)

    R = np.zeros(rot.shape + (3,3))
    
    R[...,axis,axis] = 1.0
    R[...,(axis+1)%3, (axis+1)%3] = np.cos(rot)
    R[...,(axis+1)%3, (axis+2)%3] = -np.sin(rot)
    R[...,(axis+2)%3, (axis+1)%3] = np.sin(rot)
    R[...,(axis+2)%3, (axis+2)%3] = np.cos(rot)
    return R

@jit
def get_rotation_align_z(vec):
    # rotate coordinate system to align z-axis with vec
    # first, rotate around y-axis. Then, rotate around transformed x-axis
    
    rot_y = np.arctan2(vec[...,0], vec[...,2]) # rotation direction: z->x
    rot_x = -np.arcsin(vec[...,1]) # rotation direction y->z
    
    if vec.ndim == 1:
        R_y = np.zeros((3,3))
        R_x = np.zeros((3,3))
        R_y = get_rotation_around_xyz_jit_pixel(-rot_y, axis=1)
        R_x = get_rotation_around_xyz_jit_pixel(-rot_x, axis=0)
    else:
        R_y = np.zeros(vec.shape[0:-1] + (3,3))
        R_x = np.zeros(vec.shape[0:-1] + (3,3))
        R_y = get_rotation_around_xyz_jit(-rot_y, axis=1)
        R_x = get_rotation_around_xyz_jit(-rot_x, axis=0)
    
    R_c2v = R_x @ R_y # camera CS to vec CS
    return R_c2v

# get polar coordinates of half vector and difference vector in Rusinkiewicz’s coordinate system
# cannot determine halfway vector when l and v are parallel
def Rusinkiewicz_transform(n,l,v):
    hv = (l+v)/np.linalg.norm(l+v, axis=-1, keepdims=True)
    R_c2n = get_rotation_align_z(n) # transformation into normal coordinate system (z -> n)

    clamp_cos_nh = np.clip(np.sum(n*hv, axis=-1), -1.0, 1.0)
    theta_h = np.arccos(clamp_cos_nh)
    clamp_cos_lh = np.clip(np.sum(l*hv, axis=-1), -1.0, 1.0)
    theta_d = np.arccos(clamp_cos_lh)

    '''
    Rusinkiewicz's coordinate system
    n -> z, t -> x, b -> y
    '''
    hv_n = (R_c2n @ hv[...,np.newaxis])[...,0]
    phi_h = np.arctan2(hv_n[...,1], hv_n[...,0])

    l_n = (R_c2n @ l[...,np.newaxis])[...,0]
    R_n = get_rotation_around_xyz(-phi_h, axis=2)
    R_b = get_rotation_around_xyz(-theta_h, axis=1)
    d = (R_b @ (R_n @ l_n[...,np.newaxis]))[...,0]
    phi_d = np.arctan2(d[...,1],d[...,0])
    return theta_h, theta_d, phi_d

# get normal and light direction in camera coordinate system from theta_h, theta_d, phi_d
def Rusinkiewicz2nl(theta_h, theta_d, phi_d, v_c):
    '''
    v_c: viewing direction in camera coordinate system
    '''
    # compute light direction l_n and viewing direction v_n in the coordinate system where normal is the north pole
    d = np.stack([np.cos(phi_d)*np.sin(theta_d), np.sin(phi_d)*np.sin(theta_d), np.cos(theta_d)], axis=-1)
    R_b  = get_rotation_around_xyz(theta_h, axis=1)
    l_n = (R_b @ d[...,np.newaxis])[...,0]
    hv_n = np.stack([np.sin(theta_h), np.zeros(theta_h.shape), np.cos(theta_h)], axis=-1)
    v_n = 2*np.sum(l_n*hv_n, axis=-1, keepdims=True)*hv_n - l_n
    
    # solve R_y * v_c = R_x^T * v_n, where R_c2n = R_x * R_y
    # R_x and R_y are rotation around x and y axis, respectively.
    v_cx = v_c[...,0]
    v_cy = v_c[...,1]
    v_cz = v_c[...,2]
    v_nx = v_n[...,0]
    v_ny = v_n[...,1]
    v_nz = v_n[...,2]
    alpha = np.arctan2(v_cz, v_cx)
    beta = np.arctan2(v_nz, v_ny)
    rot_y = np.arccos(v_nx/np.sqrt(v_cx**2 + v_cz**2)) - alpha
    rot_x = np.arccos(v_cy/np.sqrt(v_ny**2 + v_nz**2)) - beta

    R_y = get_rotation_around_xyz(-rot_y, axis=1)
    R_x = get_rotation_around_xyz(-rot_x, axis=0)
    R_c2n = R_x @ R_y # camera CS to normal CS
    dims_th = theta_h.ndim
    dim_offset = tuple(np.arange(0,dims_th))
    R_n2c = R_c2n.transpose(dim_offset + (dims_th + 1, dims_th))
    
    n_c = R_n2c @ np.array([0,0,1.0])
    l_c = (R_n2c @ l_n[...,np.newaxis])[...,0]
    return n_c, l_c

