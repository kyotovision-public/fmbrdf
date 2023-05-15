import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import linalg as LA

REFRACTIVE_INDEX_MIN = 1.1
REFRACTIVE_INDEX_MAX = 2.5
KAPPA_MIN = 0
KAPPA_MAX = 100
ALPHA_MIN = 0.001
ALPHA_MAX = 2*np.pi
BETA_MIN = 0.5
BETA_MAX = 4.0

# shadowing function network
class Net_Shadowing(nn.Module):
    '''
    input: alpha, beta, theta_o
    output: Lambda
    '''
    def __init__(self):
        super(Net_Shadowing, self).__init__()
        self.fc1 = nn.Linear(4, 300)
        self.fc2 = nn.Linear(300, 100)
        self.fc3 = nn.Linear(100, 50)
        self.fc4 = nn.Linear(50, 1)

    def forward(self, x):
        x[...,2] = torch.tan(x[...,2])
        a_inv = (x[...,0] * x[...,2]).unsqueeze(-1)
        x = torch.cat([x, a_inv], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x[...,0]

# body reflection network
class Net_BodyReflection(nn.Module):

    def __init__(self):
        super(Net_BodyReflection, self).__init__()
        self.fc_I1 = nn.Linear(11, 200)
        self.fc_I2 = nn.Linear(200, 100)
        self.fc_I3 = nn.Linear(100, 100)
        self.fc_I4 = nn.Linear(100, 1)
        
        self.fc_pol1 = nn.Linear(11, 200)
        self.fc_pol2 = nn.Linear(200, 100)
        self.fc_pol3 = nn.Linear(100, 100)
        self.fc_pol4 = nn.Linear(100, 2)

        self.fc_rest1 = nn.Linear(11, 200)
        self.fc_rest2 = nn.Linear(200, 100)
        self.fc_rest3 = nn.Linear(100, 100)
        self.fc_rest4 = nn.Linear(100, 6)
        
        self.act = nn.ReLU()
        self.I_act = nn.Softplus()

    def forward(self, x):
        nl = torch.sum(x[...,0:3]*x[...,3:6], dim=-1, keepdims=True) # dot(n, l)
        x = torch.cat([x, nl], dim=-1)
        device = x.device
        
        I = self.act(self.fc_I1(x))
        I = self.act(self.fc_I2(I))
        I = self.act(self.fc_I3(I))
        I = self.fc_I4(I)
        
        pol = self.act(self.fc_pol1(x))
        pol = self.act(self.fc_pol2(pol))
        pol = self.act(self.fc_pol3(pol))
        pol = self.fc_pol4(pol)

        rest = self.act(self.fc_rest1(x))
        rest = self.act(self.fc_rest2(rest))
        rest = self.act(self.fc_rest3(rest))
        rest = self.fc_rest4(rest)

        I = self.I_act(I[...,0])
        Dcos = 2 * torch.sigmoid(pol[...,0]) - 1
        Dsin = 2 * torch.sigmoid(pol[...,1]) - 1
        DoP = torch.sqrt(Dcos**2 + Dsin**2 + 1e-18)
        mask = DoP <= 1 # where not normalizing
        Dcos_norm = Dcos/(DoP + 1e-18)
        Dsin_norm = Dsin/(DoP + 1e-18)
        Dcos_norm[mask] = Dcos[mask]
        Dsin_norm[mask] = Dsin[mask]

        s0 = I
        s1 = I*Dcos_norm
        s2 = I*Dsin_norm

        shape = s0.shape
        stok = torch.stack([s0, s1, s2], dim=-1)
        out = torch.concat([stok, rest], dim=-1).reshape(shape + (3,3)).transpose(-2, -1)
        return out


def load_net(device, modelfile_body='../BRDFmodel/NN_model/body_reflection.pth', modelfile_shadowing='../BRDFmodel/NN_model/shadowing.pth'):
    # body reflection
    net_body = Net_BodyReflection()
    model_state_dict = torch.load(modelfile_body)
    net_body.load_state_dict(model_state_dict)
    net_body.to(device)
    # shadowing function
    net_shadowing = Net_Shadowing()
    model_state_dict_GAF = torch.load(modelfile_shadowing)
    net_shadowing.load_state_dict(model_state_dict_GAF)
    net_shadowing.to(device)
    return net_body, net_shadowing


def batchdot(x,y, dim=-1):
    return torch.sum(x*y, dim=dim)

def getFresnelParamsVec( v, n, refractive_index):
    A = 1/refractive_index
    cosr = torch.clamp(batchdot(v, n), 0.0, 1.0 - 1e-6)
    sinr = torch.sqrt(1-cosr**2)
    
    sint = A*sinr
    sint = torch.clamp(sint, 0.0, 1.0 - 1e-6)
    cost = torch.sqrt(1-sint**2)
    
    # reflectance of s,p-polarized    
    rs = (A*cosr-cost)/(A*cosr+cost + 1.0e-9)
    rp = -(A*cost-cosr)/(A*cost+cosr + 1.0e-9)
    
    # power reflectance
    Rp = rp**2
    Rs = rs**2
    
    # power transmittance
    Tp = 1-Rp
    Ts = 1-Rs
    
    return Rp, Rs, Tp, Ts, rp, rs


def get_rotation_around_xyz(rot, axis):
    assert (axis == 0) or (axis == 1) or (axis == 2)

    device = rot.device
    if rot.ndim == 0:
        R = torch.zeros((3,3)).float().to(device)
    else:
        R = torch.zeros(rot.size() + (3,3)).float().to(device)
    
    R[...,axis,axis] = 1.0
    R[...,(axis+1)%3, (axis+1)%3] = torch.cos(rot)
    R[...,(axis+1)%3, (axis+2)%3] = -torch.sin(rot)
    R[...,(axis+2)%3, (axis+1)%3] = torch.sin(rot)
    R[...,(axis+2)%3, (axis+2)%3] = torch.cos(rot)
    return R

def get_rotation_align_z(vec):
    # rotate coordinate system to align z-axis with vec
    # first, rotate around y-axis. Then, rotate around transformed x-axis
    
    rot_y = torch.atan2(vec[...,0], vec[...,2]) # rotation direction: z->x
    rot_x = -torch.asin(vec[...,1]) # rotation direction y->z
    
    R_y = get_rotation_around_xyz(-rot_y, axis=1)
    R_x = get_rotation_around_xyz(-rot_x, axis=0)
    
    R_c2v = R_x @ R_y # camera CS to vec CS
    return R_c2v



def separable_GAF(Lambda_i, Lambda_o):
    return 1/((1 + Lambda_i)*(1 + Lambda_o))

# compute geometric attenuation terms with a neural network
def GAFNN(n, l, v, params, net, joint_function=separable_GAF):
    shape = n.size()[0:-1]
    params_tile = torch.tile(params, shape + (1, ))

    # copmute theta_i, theta_o
    cos_i = torch.clamp(batchdot(n, l), 1e-6, 1 - 1e-6)
    cos_o = torch.clamp(batchdot(n, v), 1e-6, 1 - 1e-6)
    theta_i = torch.acos(cos_i)
    theta_o = torch.acos(cos_o)
    
    # construct input of nueral network
    xi = torch.cat((params_tile, theta_i.unsqueeze(-1)), dim=-1)
    xo = torch.cat((params_tile, theta_o.unsqueeze(-1)), dim=-1)

    Lambda_i = net(xi)
    Lambda_o = net(xo)
    return joint_function(Lambda_i, Lambda_o)

def visible_and_lit_mask(n, l, v):
    is_visible = batchdot(n, v)>0
    is_lit = batchdot(n, l)>0
    is_visible_and_lit_mask = torch.logical_and(is_visible, is_lit)
    return is_visible_and_lit_mask


def specular(slope_area_distribution, n, l, v, refractive_index, k_s, params, c):
    '''
    v-coordinate system
    '''
    device = n.device
    h = (l+v)/LA.vector_norm(l+v, dim=-1, keepdim=True)

    Rp, Rs, Tp, Ts, rp, rs = getFresnelParamsVec( v, h, refractive_index)
    R_plus = (Rs + Rp)/2.0
    R_minus = (Rs - Rp)/2.0
    phi_o = -np.pi/2.0 + torch.atan2(h[...,1], h[...,0])
    s_s = torch.stack((R_plus, R_minus*torch.cos(2*phi_o), R_minus*torch.sin(2*phi_o)), dim=-1)
    
    P = slope_area_distribution(n, h, params, c)
    
    is_visible_and_lit_mask = visible_and_lit_mask(n, l, v) # include dot(n,h) > 0
    cos_nv = torch.clamp(batchdot(n,v), 0.0, 1.0)
    cos_nh = torch.clamp(batchdot(n,h), 0.0, 1.0)
    s_s = (k_s * P * is_visible_and_lit_mask / (4*cos_nv*cos_nh + 1e-9)).unsqueeze(-1) * s_s
    return s_s

def generalized_gauss_distribution(n, a, params, c):
    cos_na = torch.clamp(batchdot(n,a), 0.0, 1.0 - 1e-6)
    theta_a = torch.acos(cos_na)
    return c*torch.exp(-(theta_a**params[1]/params[0]**params[1]))*cos_na

# compute normalization term of the microfacet distribution
def compute_c_isotropic(slope_area_distribution, params, step):
    device = params.device
    
    theta_a = torch.linspace(0, np.pi/2.0, steps=step).float().to(device)
    d_theta_a = (np.pi/2.0)/(step-1)
    
    n = torch.tensor([0,0,-1.0]).float().to(device)
    n = torch.tile(n, (step, 1))
    a = torch.stack((torch.sin(theta_a), torch.zeros(step).float().to(device), -torch.cos(theta_a)), dim=-1)
    c_inv = slope_area_distribution(n, a, params, 1.0) * torch.sin(theta_a) * d_theta_a
    c_inv = 2*np.pi*torch.sum(c_inv)
    return 1/c_inv


# preprocess: rotate normal and lighting direction
def preprocess(n, l):
    shape = n.size()[0:-1]
    device = n.device

    # rotation angle of coordinate system
    rot = torch.atan2(l[...,1],l[...,0])
    R = torch.zeros(shape + (3, 3)).float().to(device)
    R[:,0,0] = torch.cos(rot)
    R[:,0,1] = torch.sin(rot)
    R[:,1,0] = -torch.sin(rot)
    R[:,1,1] = torch.cos(rot)
    R[:,2,2] = 1.0
    ret_n = (R @ n.unsqueeze(-1))[...,0]
    ret_l = (R @ l.unsqueeze(-1))[...,0]

    return ret_n, ret_l

# postprocess: rotate AoLP
def postprocess(mueller, l):
    shape = mueller.size()[0:-2]
    device = l.device
    
    # rotation angle of coordinate system
    rot = -torch.atan2(l[...,1],l[...,0])
    C = torch.zeros(shape + (3, 3)).float().to(device)
    C[:,0,0] = 1.0
    C[:,1,1] = torch.cos(2*rot)
    C[:,1,2] = torch.sin(2*rot)
    C[:,2,1] = -torch.sin(2*rot)
    C[:,2,2] = torch.cos(2*rot)
    ret_mueller = C @ mueller
    
    return ret_mueller

def diffuse(net, n, l, v, refractive_index, rho, params, kappa):
    '''
    v-coordinate system
    '''
    shape = n.size()[0:-1]
    n_, l_ = preprocess(n, l) # rotate n, l. l_ is tiled.

    x1 = n_
    x2 = l_
    x3 = torch.tile(refractive_index, shape + (1, ))
    x4 = torch.tile(params[0], shape + (1, ))
    x5 = torch.tile(params[1], shape + (1, ))
    x6 = torch.tile(kappa, shape + (1,))
    x = torch.cat((x1,x2,x3,x4,x5,x6), dim=-1)
    outputs = net(x)
    M_b = rho * outputs
    # post process
    M_b = postprocess(M_b, l)
    s_b = M_b[...,0]

    is_visible_and_lit_mask = visible_and_lit_mask(n, l, v)
    s_b = is_visible_and_lit_mask.unsqueeze(-1) * s_b
    return s_b


# transform into v-coordinate
def transform_c2v(n, v, l):
    R_c2v = get_rotation_align_z(-v)
    ret_n = (R_c2v @ n.unsqueeze(-1))[...,0]
    ret_v = (R_c2v @ v.unsqueeze(-1))[...,0]
    ret_l = (R_c2v @ l.unsqueeze(-1))[...,0]
    return ret_n, ret_v, ret_l, R_c2v

# project Stokes vector onto an image plane
def project_stokes(s, R_c2v):
    device = s.device
    R_v2c = torch.transpose(R_c2v, -2, -1)

    I = s[...,0]
    DoLP = torch.sqrt(s[...,1]**2 + s[...,2]**2 + 1e-30)/(s[...,0] + 1e-9)
    AoLP = torch.zeros_like(I, dtype=torch.float, device=device)
    mask = DoLP > 2e-6
    AoLP[mask] = torch.atan2(s[mask,2], s[mask,1])/2
    # compute AoLP in image coordinate system
    pol_vec = torch.stack([torch.cos(AoLP), torch.sin(AoLP), torch.zeros_like(AoLP)], dim=-1)
    pol_vec = (R_v2c @ pol_vec.unsqueeze(-1))[...,0]
    AoLP_img = torch.atan2(pol_vec[...,1], pol_vec[...,0])

    # reconstruct stokes vector in image coordinate system from I, DoLP, and AoLP_img
    ret_stokes = torch.stack([I, I*DoLP*torch.cos(2*AoLP_img), I*DoLP*torch.sin(2*AoLP_img)], dim=-1)
    return ret_stokes

def specular_and_diffuse(slope_area_distribution, net, net_GAF, n, v, l, refractive_index, rho, k_s, params, kappa, c, joint_function=separable_GAF):
    device = n.device
    GAF = GAFNN(n, l, v, params, net_GAF, joint_function=joint_function)

    n_v, v_v, l_v, R_c2v = transform_c2v(n, v, l)
    s_s = specular(slope_area_distribution, n_v, l_v, v_v, refractive_index, k_s, params, c)
    s_b = diffuse(net, n_v, l_v, v_v, refractive_index, rho, params, kappa)
    ret_stok = GAF.unsqueeze(-1)*project_stokes(s_s+s_b, R_c2v)

    return ret_stok


def rendering_torch(slope_area_distribution, net, net_GAF, n, v, l_list, refractive_index, rho, k_s, params, kappa, mask_list, step_c, joint_function=separable_GAF):
    stok_multi = None
    l_num = l_list.size()[0]
    device = n.device
    
    n_clamp = n/(LA.vector_norm(n, dim=-1, keepdim=True) + 1e-9)
    v_clamp = v/(LA.vector_norm(v, dim=-1, keepdim=True) + 1e-9)
    l_list_clamp = l_list/(LA.vector_norm(l_list, dim=-1, keepdim=True) + 1e-9)
    refractive_index_clamp = torch.clamp(refractive_index, REFRACTIVE_INDEX_MIN, REFRACTIVE_INDEX_MAX)
    rho_clamp = torch.clamp(rho, min=0.0)
    k_s_clamp = torch.clamp(k_s, min=0.0)
    alpha_clamp = torch.clamp(params[0], ALPHA_MIN, ALPHA_MAX)
    beta_clamp = torch.clamp(params[1], BETA_MIN, BETA_MAX)
    params_clamp = torch.stack([alpha_clamp, beta_clamp])
    kappa_clamp = torch.clamp(kappa, KAPPA_MIN, KAPPA_MAX)

    c = compute_c_isotropic(slope_area_distribution, params_clamp, step_c)
    for l_idx in range(l_num):
        mask = mask_list[l_idx]
        mask_torch = torch.tensor(mask, dtype=bool).to(device)
        n_masked = n_clamp[mask_torch]
        v_masked = v_clamp[mask_torch]
        l = l_list_clamp[l_idx]

        # rendering
        stok = specular_and_diffuse(slope_area_distribution, net, net_GAF, n_masked, v_masked, l, refractive_index_clamp, rho_clamp, k_s_clamp, params_clamp, kappa_clamp, c, joint_function=joint_function)
        
        if stok_multi is None:
            stok_multi = stok
        else:
            stok_multi = torch.cat((stok_multi, stok), dim=0)
    return stok_multi

class Renderer:
    def __init__(self, slope_area_distribution, n, l_list, v, refractive_index, albedo_ratio, k_s, params, kappa, mask_list):
        self.l_num = len(l_list)

        self.slope_area_distribution = slope_area_distribution
        self.n = n
        self.l_list = l_list
        self.v = v
        self.refractive_index = refractive_index
        self.albedo_ratio = albedo_ratio
        self.k_s = k_s
        self.rho = self.albedo_ratio * self.k_s
        self.params = params
        self.kappa = kappa
        if len(mask_list.shape) == 2:
            mask_list = np.tile(mask_list, (self.l_num,1,1))
        self.mask_list = mask_list

        self.h = n.shape[0]
        self.w = n.shape[1]
        if len(self.v.shape) == 1:
            self.v = np.tile(self.v, (self.h, self.w, 1))

    def rendering(self, net, net_GAF, step_c, joint_function=separable_GAF):
        device = next(net.parameters()).device

        n_torch = torch.tensor(self.n, device=device, dtype=torch.float)
        v_torch = torch.tensor(self.v, device=device, dtype=torch.float)
        l_list_torch = torch.tensor(self.l_list, device=device, dtype=torch.float)
        refractive_index_torch = torch.tensor(self.refractive_index, device=device, dtype=torch.float)
        rho_torch = torch.tensor(self.rho, device=device, dtype=torch.float)
        k_s_torch = torch.tensor(self.k_s, device=device, dtype=torch.float)
        params_torch = torch.tensor(self.params, device=device, dtype=torch.float)
        kappa_torch = torch.tensor(self.kappa, device=device, dtype=torch.float)

        with torch.no_grad():
            stok_multi_torch = rendering_torch(self.slope_area_distribution, net, net_GAF, n_torch, v_torch, l_list_torch, refractive_index_torch, rho_torch, k_s_torch, params_torch, kappa_torch, self.mask_list, step_c, joint_function=joint_function)
        
        # convert to numpy
        self.stok_multi = np.zeros(self.mask_list.shape + (3,))
        self.stok_multi[self.mask_list] = stok_multi_torch.detach().cpu().numpy() 
        self.intensity_multi = self.stok_multi[...,0]
        self.DoLP_multi = np.sqrt(self.stok_multi[...,1]**2 + self.stok_multi[...,2]**2)/(np.abs(self.stok_multi[...,0]) + 1e-9)
        self.AoLP_multi = np.arctan2(self.stok_multi[...,2], self.stok_multi[...,1])/2.0


def Phi4(theta_nv):
    return np.array([theta_nv**4, theta_nv**3, theta_nv**2, theta_nv]).T

def curve_fitting_DoLP(theta_nv, DoLP, compute_Phi):
    Phi = compute_Phi(theta_nv)
    w = np.linalg.inv(Phi.T @ Phi) @ (Phi.T @ DoLP)
    return w

def robust_fitting_DoLP(theta_nv, DoLP, compute_Phi,iteration=10):
    w = None
    for i in range(iteration):
        w = curve_fitting_DoLP(theta_nv, DoLP, compute_Phi)
        r = np.abs(DoLP - compute_Phi(theta_nv) @ w)
        m = np.median(r)
        theta_nv = theta_nv[r < 6*m]
        DoLP = DoLP[r < 6*m]
    return w

def compute_DoLP_weight(mask_list, n, v, observed_DoLP, outlier, compute_Phi, iteration=10, bias_weight=0):
    l_num = len(mask_list)
    weight = np.ones(observed_DoLP.shape)
    for i in range(l_num):
        mask = mask_list[i]
        n_i = n[mask]
        v_i = v[mask]
        
        theta_nv = (180/np.pi)*np.arccos(np.sum(n_i*v_i, axis=-1))
        DoLP = observed_DoLP[i, mask]

        w = robust_fitting_DoLP(theta_nv, DoLP, compute_Phi, iteration)
        DoLP_fitting = compute_Phi(theta_nv) @ w
        residual = DoLP - DoLP_fitting
        n_outlier = np.sum((residual >= outlier))
        weight[i, mask][residual >= outlier] = (np.sum(mask) - n_outlier)/(n_outlier + bias_weight + 1e-9)
    
    return weight[mask_list]


# DoLP (no weight)
# DoLP fitting
class mse_loss_DoLP:
    def __init__(self, loss=nn.HuberLoss(reduction='mean')):
        self.loss = loss
    
    def __call__(self, s_pred, s_target):
        DoLP_pred = torch.sqrt(s_pred[...,1]**2+s_pred[...,2]**2 + 1e-30)/(s_pred[...,0] + 1e-9)
        DoLP_target = torch.sqrt(s_target[...,1]**2+s_target[...,2]**2 + 1e-30)/(s_target[...,0] + 1e-9)
        return self.loss(DoLP_pred, DoLP_target)


# weighted DoLP
# BRDF estimation loss
class weighted_mse_loss_DoLP:
    def __init__(self, weight, device, loss=nn.HuberLoss(reduction='sum')):
        self.weight = torch.tensor(weight, dtype=torch.float, device=device)
        self.loss = loss
    
    def __call__(self, s_pred, s_target):
        DoLP_pred = torch.sqrt(s_pred[...,1]**2+s_pred[...,2]**2 + 1e-30)/(s_pred[...,0] + 1e-9)
        DoLP_target = torch.sqrt(s_target[...,1]**2+s_target[...,2]**2 + 1e-30)/(s_target[...,0] + 1e-9)
        sqrt_w = torch.sqrt(self.weight)
        return self.loss(sqrt_w*DoLP_pred, sqrt_w*DoLP_target)/torch.sum(self.weight)

# intensity + weighted DoLP
# BRDF estimation loss
class weighted_mse_loss_intensity_DoLP:
    def __init__(self, weight, device, lam, loss1=nn.HuberLoss(reduction='sum'), loss2=nn.HuberLoss(reduction='sum')):
        self.weight = torch.tensor(weight, dtype=torch.float, device=device)
        self.lam = lam
        self.loss1 = loss1
        self.loss2 = loss2
    
    def __call__(self, s_pred, s_target):
        intensity_loss = self.loss1(s_pred[...,0], s_target[...,0])/torch.numel(s_pred[...,0])

        DoLP_pred = torch.sqrt(s_pred[...,1]**2+s_pred[...,2]**2 + 1e-30)/(s_pred[...,0] + 1e-9)
        DoLP_target = torch.sqrt(s_target[...,1]**2+s_target[...,2]**2 + 1e-30)/(s_target[...,0] + 1e-9)
        sqrt_w = torch.sqrt(self.weight)
        DoLP_loss = self.loss2(sqrt_w*DoLP_pred, sqrt_w*DoLP_target)/torch.sum(self.weight)
        return intensity_loss + self.lam*DoLP_loss


# intensity + DoLP
# BRDF estimation during alternating optimization
class mse_loss_intensity_DoLP:
    def __init__(self, lam, loss1=nn.HuberLoss(reduction='mean'), loss2=nn.HuberLoss(reduction='mean')):
        self.lam = lam
        self.loss1 = loss1
        self.loss2 = loss2
        
    def __call__(self, s_pred, s_target):
        intensity_loss = self.loss1(s_pred[...,0], s_target[...,0])

        DoLP_pred = torch.sqrt(s_pred[...,1]**2+s_pred[...,2]**2 + 1e-30)/(s_pred[...,0] + 1e-9)
        DoLP_target = torch.sqrt(s_target[...,1]**2+s_target[...,2]**2 + 1e-30)/(s_target[...,0] + 1e-9)
        DoLP_loss = self.loss2(DoLP_pred, DoLP_target)
        
        return intensity_loss + self.lam*DoLP_loss

# intensity
# Photometric Stereo loss
class mse_loss_intensity:
    def __init__(self, loss=nn.HuberLoss(reduction='mean')):
        self.loss = loss
        
    def __call__(self, s_pred, s_target):
        return self.loss(s_pred[...,0], s_target[...,0])

# Stokes
# Photometric Stereo stok loss
class mse_loss_stok:
    def __init__(self, lam, loss1=nn.HuberLoss(reduction='mean'), loss2=nn.HuberLoss(reduction='mean')):
        self.loss1 = loss1
        self.loss2 = loss2
        self.lam = lam
        
    def __call__(self, s_pred, s_target):
        intensity_loss = self.loss1(s_pred[...,0], s_target[...,0])
        polarization_loss = self.loss2(s_pred[...,1:], s_target[...,1:])
        return intensity_loss + self.lam*polarization_loss

# polarization
# Photometric Stereo polarization loss
class mse_loss_polarization:
    def __init__(self, loss=nn.HuberLoss(reduction='mean')):
        self.loss = loss
        
    def __call__(self, s_pred, s_target):
        I_pred = s_pred[...,0]
        I_target = s_target[...,0]
        polarization_loss = self.loss(s_pred[...,1:]/(I_pred[...,np.newaxis] + 1e-9), s_target[...,1:]/(I_target[...,np.newaxis] + 1e-9))
        return polarization_loss

# intensity + polarization
# Photometric Stereo intensity polarization loss
class mse_loss_intensity_polarization:
    def __init__(self, lam, loss1=nn.HuberLoss(reduction='mean'), loss2=nn.HuberLoss(reduction='mean')):
        self.lam = lam
        self.loss1 = loss1
        self.loss2 = loss2

    def __call__(self, s_pred, s_target):
        I_pred = s_pred[...,0]
        I_target = s_target[...,0]
        intensity_loss = self.loss1(I_pred, I_target)

        polarization_loss = self.loss2(s_pred[...,1:]/(I_pred[...,np.newaxis] + 1e-9), s_target[...,1:]/(I_target[...,np.newaxis] + 1e-9))
        return intensity_loss + self.lam*polarization_loss


# intensity + DoLP + Stokes
# joint optimization loss
class mse_loss_intensity_DoLP_stok:
    def __init__(self, lam1, lam2, loss1=nn.HuberLoss(reduction='mean'), loss2=nn.HuberLoss(reduction='mean'), loss3=nn.HuberLoss(reduction='mean')):
        self.lam1 = lam1
        self.lam2 = lam2
        self.loss1 = loss1
        self.loss2 = loss2
        self.loss3 = loss3

    def __call__(self, s_pred, s_target):
        intensity_loss = self.loss1(s_pred[...,0], s_target[...,0])

        DoLP_pred = torch.sqrt(s_pred[...,1]**2+s_pred[...,2]**2 + 1e-30)/(s_pred[...,0] + 1e-9)
        DoLP_target = torch.sqrt(s_target[...,1]**2+s_target[...,2]**2 + 1e-30)/(s_target[...,0] + 1e-9)
        DoLP_loss = self.loss2(DoLP_pred, DoLP_target)

        stok_loss = self.loss3(s_pred[...,1:], s_target[...,1:]) # AoLP loss
        return intensity_loss + self.lam1*DoLP_loss + self.lam2*stok_loss

class smoothness_loss:
    def __init__(self, window_radius, lam):
        self.r = window_radius
        self.lam = lam
    
    def __call__(self, n, mask):
        if self.lam <= 0:
            return 0.0

        device = n.device
        n_clamp = n/(LA.vector_norm(n, dim=-1, keepdim=True) + 1e-9)
        n_padding = F.pad(n_clamp.permute(2,0,1), pad=(self.r,self.r,self.r,self.r), mode="replicate").permute(1,2,0)
        mask_torch = torch.tensor(mask, device=device, dtype=bool, requires_grad=False)
        mask_padding = F.pad(mask_torch, pad=(self.r,self.r,self.r,self.r), mode="constant", value=False)

        roll_shifts_list = []
        for i in range(-self.r, self.r+1):
            for j in range(-self.r, self.r+1):
                if (i > 0) or ((i == 0) and (j > 0)):
                    roll_shifts_list.append((i,j))

        loss_list = torch.zeros(len(roll_shifts_list), device=device, dtype=torch.float)
        N = 0
        for i, shifts in enumerate(roll_shifts_list):
            n_shifted = torch.roll(n_padding, shifts=shifts, dims=(0,1))
            mask_shifted = torch.roll(mask_padding, shifts=shifts, dims=(0,1))
            mask_temp = (mask_torch & mask_shifted[self.r:-self.r,self.r:-self.r])

            loss_list[i] = torch.sum((1 - batchdot(n_clamp, n_shifted[self.r:-self.r,self.r:-self.r])[mask_temp])/2.0)
            N += torch.sum(mask_temp)

        loss = torch.sum(loss_list)/N
        return self.lam * loss

class Optimizer:
    def __init__(self, slope_area_distribution, observed_stok, n, l_list, v, refractive_index, albedo_ratio, k_s, params, kappa, mask_list, net, net_GAF, joint_function=separable_GAF):
        assert len(l_list) == len(observed_stok), 'shape inconsistency between l_list and observed_stok'
        
        h = n.shape[0]
        w = n.shape[1]
        if len(v.shape) == 1:
            v = np.tile(v, (h, w, 1))
        
        device = next(net.parameters()).device
        self.net = net
        self.net_GAF = net_GAF
        self.joint_function = joint_function
        self.slope_area_distribution = slope_area_distribution
        self.mask_list = mask_list

        # convert into tensor
        self.n_torch = torch.tensor(n, device=device, dtype=torch.float)
        self.v_torch = torch.tensor(v, device=device, dtype=torch.float)
        self.l_list_torch = torch.tensor(l_list, device=device, dtype=torch.float)
        self.refractive_index_torch = torch.tensor(refractive_index, device=device, dtype=torch.float)
        self.albedo_ratio_torch = torch.tensor(albedo_ratio, device=device, dtype=torch.float)
        self.k_s_torch = torch.tensor(k_s, device=device, dtype=torch.float)
        self.params_torch = torch.tensor(params, device=device, dtype=torch.float)
        self.kappa_torch = torch.tensor(kappa, device=device, dtype=torch.float)
        
        self.observed_stok_torch = torch.tensor(observed_stok[self.mask_list], device=device, dtype=torch.float)

    def optimize(self, optimizer, loss_fn, step_c, iteration, epsilon=1e-6, isPrint=True, loss_smooth=None):
    
        old_loss = 0.0
        for i in range(iteration):
            optimizer.zero_grad()

            self.rho_torch = self.albedo_ratio_torch * self.k_s_torch
            stok_multi_torch = rendering_torch(self.slope_area_distribution, self.net, self.net_GAF, self.n_torch, self.v_torch, self.l_list_torch, self.refractive_index_torch, self.rho_torch, self.k_s_torch, self.params_torch, self.kappa_torch, self.mask_list, step_c, joint_function=self.joint_function)
            
            if loss_smooth is not None:
                mask = np.any(self.mask_list, axis=0)
                loss = loss_fn(stok_multi_torch, self.observed_stok_torch) + loss_smooth(self.n_torch, mask)
            else:
                loss = loss_fn(stok_multi_torch, self.observed_stok_torch)
            loss.backward()
            
            if isPrint:
                print(f"step: {i}, loss: {loss.item()}")
            
            if np.abs(old_loss-loss.item())/loss.item() < epsilon:
                print('complete')
                break
            optimizer.step()
            self.n_torch.data = self.n_torch.data/(LA.vector_norm(self.n_torch.data, dim=-1, keepdim=True) + 1e-9)
            self.v_torch.data = self.v_torch.data/(LA.vector_norm(self.v_torch.data, dim=-1, keepdim=True) + 1e-9)
            self.l_list_torch.data = self.l_list_torch.data/(LA.vector_norm(self.l_list_torch.data, dim=-1, keepdim=True) + 1e-9)
            self.refractive_index_torch.data = torch.clamp(self.refractive_index_torch.data, REFRACTIVE_INDEX_MIN, REFRACTIVE_INDEX_MAX)
            self.albedo_ratio_torch.data = torch.clamp(self.albedo_ratio_torch.data, min=0.0)
            self.k_s_torch.data = torch.clamp(self.k_s_torch.data, min=0.0)
            alpha_clamp = torch.clamp(self.params_torch[0].data, ALPHA_MIN, ALPHA_MAX)
            beta_clamp = torch.clamp(self.params_torch[1].data, BETA_MIN, BETA_MAX)
            self.params_torch.data = torch.stack([alpha_clamp, beta_clamp])
            self.kappa_torch.data = torch.clamp(self.kappa_torch.data, KAPPA_MIN, KAPPA_MAX)
            
            old_loss = loss.item()

    def numpy(self):
        n_norm = self.n_torch/(LA.vector_norm(self.n_torch, dim=-1, keepdim=True) + 1e-9)
        v_norm = self.v_torch/(LA.vector_norm(self.v_torch, dim=-1, keepdim=True) + 1e-9)
        l_list_norm = self.l_list_torch/(LA.vector_norm(self.l_list_torch, dim=-1, keepdim=True) + 1e-9)
    
        n = n_norm.detach().cpu().numpy()
        v = v_norm.detach().cpu().numpy()
        l_list = l_list_norm.detach().cpu().numpy()
        refractive_index = self.refractive_index_torch.detach().cpu().numpy()
        albedo_ratio = self.albedo_ratio_torch.detach().cpu().numpy()
        k_s = self.k_s_torch.detach().cpu().numpy()
        params = self.params_torch.detach().cpu().numpy()
        kappa = self.kappa_torch.detach().cpu().numpy()
        return n, v, l_list, refractive_index, albedo_ratio, k_s, params, kappa


def BRDFestimation(slope_area_distribution, observed_stok, outlier, compute_Phi, n, l_list, v, refractive_index_init, albedo_ratio_init, k_s_init, params_init, kappa_init, mask_list, step_c, net, net_GAF, lr=0.01, iteration=10000, bias_weight=0, lam=1.0, delta_huber_intensity=1.0, delta_huber_DoLP=1.0, epsilon=1e-6, isPrint=True, joint_function=separable_GAF):
    device = next(net.parameters()).device
    opt = Optimizer(slope_area_distribution, observed_stok, n, l_list, v, refractive_index_init, albedo_ratio_init, k_s_init, params_init, kappa_init, mask_list, net, net_GAF, joint_function=joint_function)
    
    observed_DoLP = np.sqrt(observed_stok[...,1]**2 + observed_stok[...,2]**2 + 1e-30)/(observed_stok[...,0] + 1e-9)
    weight = compute_DoLP_weight(mask_list, n, v, observed_DoLP, outlier, compute_Phi, iteration=10, bias_weight=bias_weight)
    loss_fn = weighted_mse_loss_intensity_DoLP(weight, device, lam, loss1=nn.HuberLoss(reduction='sum', delta=delta_huber_DoLP), loss2=nn.HuberLoss(reduction='sum', delta=delta_huber_intensity))
    opt.refractive_index_torch.requires_grad_()
    opt.albedo_ratio_torch.requires_grad_()
    opt.k_s_torch.requires_grad_()
    opt.params_torch.requires_grad_()
    opt.kappa_torch.requires_grad_()
    optimizer = optim.Adam([opt.refractive_index_torch, opt.albedo_ratio_torch, opt.k_s_torch, opt.params_torch, opt.kappa_torch], lr=lr)
    opt.optimize(optimizer, loss_fn, step_c, iteration=iteration, epsilon=epsilon, isPrint=isPrint)
    _,_,_, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est = opt.numpy()
    return refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est

def BRDFestimation_noWeight(slope_area_distribution, observed_stok, n, l_list, v, refractive_index_init, albedo_ratio_init, k_s, params_init, kappa_init, mask_list, step_c, net, net_GAF, loss_fn, opt_ks, lr=0.01, iteration=10000, epsilon=1e-6, isPrint=True, joint_function=separable_GAF):
    '''
    BRDF estimation for joint estimation or BRDF fitting
    '''
    device = next(net.parameters()).device
    opt = Optimizer(slope_area_distribution, observed_stok, n, l_list, v, refractive_index_init, albedo_ratio_init, k_s, params_init, kappa_init, mask_list, net, net_GAF, joint_function=joint_function)
    
    opt.refractive_index_torch.requires_grad_()
    opt.albedo_ratio_torch.requires_grad_()
    opt.params_torch.requires_grad_()
    opt.kappa_torch.requires_grad_()
    if opt_ks:
        opt.k_s_torch.requires_grad_()
        optimizer = optim.Adam([opt.refractive_index_torch, opt.albedo_ratio_torch, opt.k_s_torch, opt.params_torch, opt.kappa_torch], lr=lr)
    else:
        optimizer = optim.Adam([opt.refractive_index_torch, opt.albedo_ratio_torch, opt.params_torch, opt.kappa_torch], lr=lr)
    opt.optimize(optimizer, loss_fn, step_c, iteration=iteration, epsilon=epsilon, isPrint=isPrint)
    _,_,_, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est = opt.numpy()
    return refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est

def PS(slope_area_distribution, observed_stok, n_init, l_list, v, refractive_index, albedo_ratio, k_s_init, params, kappa, mask_list, step_c, net, net_GAF, loss_fn, lr=0.01, iteration=10000, epsilon=1e-6, isPrint=True, joint_function=separable_GAF, loss_smooth=None):
    opt = Optimizer(slope_area_distribution, observed_stok, n_init, l_list, v, refractive_index, albedo_ratio, k_s_init, params, kappa, mask_list, net, net_GAF, joint_function=joint_function)

    opt.n_torch.requires_grad_()
    optimizer = optim.Adam([opt.n_torch], lr=lr)
    opt.optimize(optimizer, loss_fn, step_c, iteration=iteration, epsilon=epsilon, isPrint=isPrint, loss_smooth=loss_smooth)
    n_est, _, _, _, _, _, _, _ = opt.numpy()
    return n_est

def jointOptimization(slope_area_distribution, observed_stok, n_init, l_list, v, refractive_index_init, albedo_ratio_init, k_s_init, params_init, kappa_init, mask_list, step_c, net, net_GAF, loss_fn, lr=0.01, iteration=10000, epsilon=1e-6, isPrint=True, joint_function=separable_GAF, loss_smooth=None):
    opt = Optimizer(slope_area_distribution, observed_stok, n_init, l_list, v, refractive_index_init, albedo_ratio_init, k_s_init, params_init, kappa_init, mask_list, net, net_GAF, joint_function=joint_function)

    opt.n_torch.requires_grad_()
    opt.refractive_index_torch.requires_grad_()
    opt.albedo_ratio_torch.requires_grad_()
    opt.k_s_torch.requires_grad_()
    opt.params_torch.requires_grad_()
    opt.kappa_torch.requires_grad_()
    optimizer = optim.Adam([opt.n_torch, opt.refractive_index_torch, opt.albedo_ratio_torch, opt.k_s_torch, opt.params_torch, opt.kappa_torch], lr=lr)
    opt.optimize(optimizer, loss_fn, step_c, iteration=iteration, epsilon=epsilon, isPrint=isPrint, loss_smooth=loss_smooth)
    n_est,_,_, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est = opt.numpy()
    return n_est, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est

def alternating_optimization(slope_area_distribution, observed_stok, n_init, l_list, v, refractive_index_init, albedo_ratio_init, k_s_init, params_init, kappa_init, mask_list, step_c, net, net_GAF, loss_fn_BRDF, loss_fn_PS, loss_fn_joint, loss_fn_alter, opt_ks, lr=0.001, iteration_each=5000, iteration_joint=10000, max_alternating=100, epsilon=1e-6, epsilon_alter=1e-4, isPrint=True, joint_function=separable_GAF, loss_smooth_list=[None], loss_smooth_joint=None):
    device = next(net.parameters()).device
    n_est = n_init
    refractive_index_est = refractive_index_init
    albedo_ratio_est = albedo_ratio_init
    k_s_est = k_s_init
    params_est = params_init
    kappa_est = kappa_init
    old_loss = 1e9
    new_loss = 1e8
    i = 0
    while((np.abs(old_loss - new_loss)/old_loss > epsilon_alter) and (i < max_alternating)):
        print(f"iteration: {i}")
        i_loss_smooth = np.clip(i, 0, len(loss_smooth_list) -1)
        loss_smooth = loss_smooth_list[i_loss_smooth]

        refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est = BRDFestimation_noWeight(slope_area_distribution, observed_stok, n_est, l_list, v, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est, mask_list, step_c, net, net_GAF, loss_fn_BRDF, opt_ks, lr=lr, iteration=iteration_each, epsilon=epsilon, isPrint=isPrint, joint_function=joint_function)

        n_est = PS(slope_area_distribution, observed_stok, n_est, l_list, v, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est, mask_list, step_c, net, net_GAF, loss_fn_PS, lr=lr, iteration=iteration_each, epsilon=epsilon, isPrint=isPrint, joint_function=joint_function, loss_smooth=loss_smooth)

        renderer = Renderer(slope_area_distribution, n_est, l_list, v, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est, mask_list)
        renderer.rendering(net, net_GAF, step_c, joint_function=joint_function)
        # update loss
        old_loss = new_loss
        stok_multi_torch = torch.tensor(renderer.stok_multi[mask_list], device=device, dtype=torch.float, requires_grad=False)

        observed_stok_torch = torch.tensor(observed_stok[mask_list], device=device, dtype=torch.float, requires_grad=False)
        new_loss = loss_fn_alter(stok_multi_torch, observed_stok_torch).item()
        print(f'old loss {i}: {old_loss}')
        print(f'loss {i}: {new_loss}')
        i += 1
    print(f'number of alternating optimization {i}')
    print("Joint optimization")
    n_est, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est = jointOptimization(slope_area_distribution, observed_stok, n_est, l_list, v, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est, mask_list, step_c, net, net_GAF, loss_fn_joint, lr=lr, iteration=iteration_joint, epsilon=epsilon, isPrint=isPrint, joint_function=joint_function, loss_smooth=loss_smooth_joint)
    return n_est, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est


# intensity + DoLP + Stokes
# joint optimization loss
class mse_loss_intensity_DoLP_stok_pixel:
    def __init__(self, lam1, lam2, loss1=nn.HuberLoss(reduction='none'), loss2=nn.HuberLoss(reduction='none'), loss3=nn.HuberLoss(reduction='none')):
        self.lam1 = lam1
        self.lam2 = lam2
        self.loss1 = loss1
        self.loss2 = loss2
        self.loss3 = loss3

    def __call__(self, s_pred, s_target, mask_list):
        intensity_loss = self.loss1(s_pred[...,0], s_target[...,0])

        DoLP_pred = torch.sqrt(s_pred[...,1]**2+s_pred[...,2]**2 + 1e-30)/(s_pred[...,0] + 1e-9)
        DoLP_target = torch.sqrt(s_target[...,1]**2+s_target[...,2]**2 + 1e-30)/(s_target[...,0] + 1e-9)
        DoLP_loss = self.loss2(DoLP_pred, DoLP_target)

        stok_loss = self.loss3(s_pred[...,1:], s_target[...,1:]) # AoLP loss
        stok_loss = torch.sum(stok_loss, dim=-1)
        return torch.sum((intensity_loss + self.lam1*DoLP_loss + self.lam2*stok_loss)*mask_list, dim=0)/(torch.sum(mask_list, dim=0) + 1e-9)

# intensity + polarization
# Photometric Stereo intensity polarization loss
class mse_loss_intensity_polarization_pixel:
    def __init__(self, lam, loss1=nn.HuberLoss(reduction='none'), loss2=nn.HuberLoss(reduction='none')):
        self.lam = lam
        self.loss1 = loss1
        self.loss2 = loss2

    def __call__(self, s_pred, s_target, mask_list):
        I_pred = s_pred[...,0]
        I_target = s_target[...,0]
        intensity_loss = self.loss1(I_pred, I_target)

        polarization_loss = self.loss2(s_pred[...,1:]/(I_pred[...,np.newaxis] + 1e-9), s_target[...,1:]/(I_target[...,np.newaxis] + 1e-9))
        polarization_loss = torch.sum(polarization_loss, dim=-1)
        return torch.sum((intensity_loss + self.lam*polarization_loss)*mask_list, dim=0)/(torch.sum(mask_list, dim=0) + 1e-9)

# intensity
class mse_loss_intensity_pixel:
    def __init__(self, loss=nn.HuberLoss(reduction='none')):
        self.loss = loss

    def __call__(self, s_pred, s_target, mask_list):
        intensity_loss = self.loss(s_pred[...,0], s_target[...,0])
        return torch.sum(intensity_loss*mask_list, dim=0)/(torch.sum(mask_list, dim=0) + 1e-9)

def propagate_normal(slope_area_distribution, observed_stok, n, l_list, v, refractive_index, albedo_ratio, k_s, params, kappa, mask_list, step_c, net, net_GAF, loss_fn_pixel):
    '''
    update normal by propagating reliable normal
    update all normals simultaneously
    '''
    assert len(n.shape) == 3
    device = next(net.parameters()).device
    h,w,_ = n.shape
    n = np.nan_to_num(n, 0.0)

    with torch.no_grad():
        n_torch = torch.tensor(n, device=device, dtype=torch.float, requires_grad=False)
        l_list_torch = torch.tensor(l_list, device=device, dtype=torch.float, requires_grad=False)
        v_torch = torch.tensor(v, device=device, dtype=torch.float, requires_grad=False)
        refractive_index_torch = torch.tensor(refractive_index, device=device, dtype=torch.float, requires_grad=False)
        albedo_ratio_torch = torch.tensor(albedo_ratio, device=device, dtype=torch.float, requires_grad=False)
        k_s_torch = torch.tensor(k_s, device=device, dtype=torch.float, requires_grad=False)
        params_torch = torch.tensor(params, device=device, dtype=torch.float, requires_grad=False)
        kappa_torch = torch.tensor(kappa, device=device, dtype=torch.float, requires_grad=False)
        observed_stok_torch = torch.tensor(observed_stok, device=device, dtype=torch.float, requires_grad=False)
        mask_list_torch = torch.tensor(mask_list, device=device, dtype=bool)
        rho_torch = albedo_ratio_torch*k_s_torch

        flag = True
        while(flag):
            # rendering
            loss_maps = torch.zeros((5,h,w), device=device, dtype=torch.float) # (current normal + neighbor normal)

            roll_shifts_list = ((0,0), (-1,0), (1,0), (0,-1), (0,1))
            for i, shifts in enumerate(roll_shifts_list):
                n_shifted = torch.roll(n_torch, shifts=shifts, dims=(0,1))
                stok_multi_temp = rendering_torch(slope_area_distribution, net, net_GAF, n_shifted, v_torch, l_list_torch, refractive_index_torch, rho_torch, k_s_torch, params_torch, kappa_torch, mask_list, step_c)        
                rendered_stok = torch.zeros(mask_list.shape + (3,), device=device, dtype=torch.float)
                rendered_stok[mask_list_torch] = stok_multi_temp
                loss_maps[i] = loss_fn_pixel(rendered_stok, observed_stok_torch, mask_list_torch)
                if i != 0:
                    loss_maps[i] *= 1.001

            min_indices = torch.argmin(loss_maps, dim=0)
            n_old = n_torch.clone()
            for i, shifts in enumerate(roll_shifts_list):
                update_mask = (min_indices == i)

                if i == 0:
                    continue
                n_shifted = torch.roll(n_old, shifts=shifts, dims=(0,1))
                n_torch[update_mask] = n_shifted[update_mask]

            update_count = torch.sum(min_indices > 0).detach().cpu().numpy() # the number of updated normal
            print('update pixel: ', update_count)
            flag = update_count > 0

        ret_n = n_torch.detach().cpu().numpy()
    return ret_n

def alternating_optimization_propagate(slope_area_distribution, observed_stok, n_init, l_list, v, refractive_index_init, albedo_ratio_init, k_s_init, params_init, kappa_init, mask_list, step_c, net, net_GAF, loss_fn_BRDF, loss_fn_PS, loss_fn_joint, loss_fn_alter, loss_fn_pixel, opt_ks, lr=0.001, iteration_each=5000, iteration_joint=10000, max_alternating=100, max_propagate=100, epsilon=1e-6, epsilon_alter=1e-4, epsilon_propagate=1e-4, isPrint=True, loss_smooth_list=[None], loss_smooth_joint=None, loss_smooth_joint_list=[None]):
    device = next(net.parameters()).device

    n_est, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est = alternating_optimization(slope_area_distribution, observed_stok, n_init, l_list, v, refractive_index_init, albedo_ratio_init, k_s_init, params_init, kappa_init, mask_list, step_c, net, net_GAF, loss_fn_BRDF, loss_fn_PS, loss_fn_joint, loss_fn_alter, opt_ks, lr=lr, iteration_each=iteration_each, iteration_joint=iteration_joint, max_alternating=max_alternating, epsilon=epsilon, epsilon_alter=epsilon_alter, isPrint=True, loss_smooth_list=loss_smooth_list, loss_smooth_joint=loss_smooth_joint)

    old_loss = 1e9
    new_loss = 1e8
    i = 0
    while((np.abs(old_loss - new_loss)/old_loss > epsilon_propagate) and (i < max_propagate)):
        print(f"propagation: {i}")
        i_loss_smooth_joint = np.clip(i, 0, len(loss_smooth_joint_list) -1)
        loss_smooth_joint_i = loss_smooth_joint_list[i_loss_smooth_joint]

        # propagation
        n_est = propagate_normal(slope_area_distribution, observed_stok, n_est, l_list, v, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est, mask_list, step_c, net, net_GAF, loss_fn_pixel)
        # joint optimization
        n_est, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est = jointOptimization(slope_area_distribution, observed_stok, n_est, l_list, v, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est, mask_list, step_c, net, net_GAF, loss_fn_joint, lr=lr, iteration=iteration_joint, epsilon=epsilon, isPrint=isPrint, loss_smooth=loss_smooth_joint_i)

        renderer = Renderer(slope_area_distribution, n_est, l_list, v, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est, mask_list)
        renderer.rendering(net, net_GAF, step_c)
        # update loss
        old_loss = new_loss
        stok_multi_torch = torch.tensor(renderer.stok_multi[mask_list], device=device, dtype=torch.float, requires_grad=False)

        observed_stok_torch = torch.tensor(observed_stok[mask_list], device=device, dtype=torch.float, requires_grad=False)
        new_loss = loss_fn_alter(stok_multi_torch, observed_stok_torch).item()
        print(f'old loss {i}: {old_loss}')
        print(f'loss {i}: {new_loss}')
        i += 1
    print(f'number of alternating optimization {i}')
    print("Joint optimization")
    return n_est, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est


# fitting to plane with rough mesogeometry --------------------------------------------------------------------
def rendering_plane_torch(slope_area_distribution, net, net_GAF, n, v, l, refractive_index, rho, k_s, params, kappa, step_c, joint_function=separable_GAF):
    '''
    l is not list
    return Stokes vector
    '''
    device = n.device
    
    n_clamp = n/(LA.vector_norm(n, dim=-1, keepdim=True) + 1e-9)
    v_clamp = v/(LA.vector_norm(v, dim=-1, keepdim=True) + 1e-9)
    l_clamp = l/(LA.vector_norm(l, dim=-1, keepdim=True) + 1e-9)
    refractive_index_clamp = torch.clamp(refractive_index, REFRACTIVE_INDEX_MIN, REFRACTIVE_INDEX_MAX)
    rho_clamp = torch.clamp(rho, min=0.0)
    k_s_clamp = torch.clamp(k_s, min=0.0)
    alpha_clamp = torch.clamp(params[0], ALPHA_MIN, ALPHA_MAX)
    beta_clamp = torch.clamp(params[1], BETA_MIN, BETA_MAX)
    params_clamp = torch.stack([alpha_clamp, beta_clamp])
    kappa_clamp = torch.clamp(kappa, KAPPA_MIN, KAPPA_MAX)

    c = compute_c_isotropic(slope_area_distribution, params_clamp, step_c)

    # rendering
    stok = specular_and_diffuse(slope_area_distribution, net, net_GAF, n_clamp, v_clamp, l_clamp, refractive_index_clamp, rho_clamp, k_s_clamp, params_clamp, kappa_clamp, c, joint_function=joint_function)
    
    return stok


class Renderer_plane:
    def __init__(self, slope_area_distribution, n, l, v, refractive_index, albedo_ratio, k_s, params, kappa):

        self.slope_area_distribution = slope_area_distribution
        self.n = n
        self.l = l
        self.v = v
        self.refractive_index = refractive_index
        self.albedo_ratio = albedo_ratio
        self.k_s = k_s
        self.rho = self.albedo_ratio * self.k_s
        self.params = params
        self.kappa = kappa

        self.num_img = n.shape[0]
        if len(self.v.shape) == 1:
            self.v = np.tile(self.v, (self.num_img, 1))

    def rendering(self, net, net_GAF, step_c, joint_function=separable_GAF):
        device = next(net.parameters()).device

        n_torch = torch.tensor(self.n, device=device, dtype=torch.float)
        v_torch = torch.tensor(self.v, device=device, dtype=torch.float)
        l_torch = torch.tensor(self.l, device=device, dtype=torch.float)
        refractive_index_torch = torch.tensor(self.refractive_index, device=device, dtype=torch.float)
        rho_torch = torch.tensor(self.rho, device=device, dtype=torch.float)
        k_s_torch = torch.tensor(self.k_s, device=device, dtype=torch.float)
        params_torch = torch.tensor(self.params, device=device, dtype=torch.float)
        kappa_torch = torch.tensor(self.kappa, device=device, dtype=torch.float)

        with torch.no_grad():
            stok_torch = rendering_plane_torch(self.slope_area_distribution, net, net_GAF, n_torch, v_torch, l_torch, refractive_index_torch, rho_torch, k_s_torch, params_torch, kappa_torch, step_c, joint_function=joint_function)
        
        # convert to numpy
        self.stok = stok_torch.detach().cpu().numpy()
        self.intensity = self.stok[...,0]
        self.DoLP = np.sqrt(self.stok[...,1]**2 + self.stok[...,2]**2)/(np.abs(self.stok[...,0]) + 1e-9)
        self.AoLP = np.arctan2(self.stok[...,2], self.stok[...,1])/2.0


# fitting with plane image
class Optimizer_plane:
    def __init__(self, slope_area_distribution, observed_stok, n, l, v, refractive_index, albedo_ratio, k_s, params, kappa, net, net_GAF, joint_function=separable_GAF):
       
        num_img = observed_stok.shape[0]
        if len(v.shape) == 1:
            v = np.tile(v, (num_img,1))
        
        device = next(net.parameters()).device
        self.net = net
        self.net_GAF = net_GAF
        self.joint_function = joint_function
        self.slope_area_distribution = slope_area_distribution

        # convert into tensor
        self.n_torch = torch.tensor(n, device=device, dtype=torch.float)
        self.v_torch = torch.tensor(v, device=device, dtype=torch.float)
        self.l_torch = torch.tensor(l, device=device, dtype=torch.float)
        self.refractive_index_torch = torch.tensor(refractive_index, device=device, dtype=torch.float)
        self.albedo_ratio_torch = torch.tensor(albedo_ratio, device=device, dtype=torch.float)
        self.k_s_torch = torch.tensor(k_s, device=device, dtype=torch.float)
        self.params_torch = torch.tensor(params, device=device, dtype=torch.float)
        self.kappa_torch = torch.tensor(kappa, device=device, dtype=torch.float)
        
        self.rho_torch = self.albedo_ratio_torch * self.k_s_torch
        self.observed_stok_torch = torch.tensor(observed_stok, device=device, dtype=torch.float)


    def optimize(self, optimizer, loss_fn, step_c, iteration, epsilon=1e-6, isPrint=True):
        old_loss = 0.0
        for i in range(iteration):
            optimizer.zero_grad()

            self.rho_torch = self.albedo_ratio_torch * self.k_s_torch
            stok_torch = rendering_plane_torch(self.slope_area_distribution, self.net, self.net_GAF, self.n_torch, self.v_torch, self.l_torch, self.refractive_index_torch, self.rho_torch, self.k_s_torch, self.params_torch, self.kappa_torch, step_c, joint_function=self.joint_function)
            
            loss = loss_fn(stok_torch, self.observed_stok_torch)
            loss.backward()
            if isPrint:
                print(f"step: {i}, loss: {loss.item()}")
            if np.abs(old_loss-loss.item())/loss.item() < epsilon:
                print('complete')
                break
            optimizer.step()

            self.n_torch.data = self.n_torch.data/(LA.vector_norm(self.n_torch.data, dim=-1, keepdim=True) + 1e-9)
            self.v_torch.data = self.v_torch.data/(LA.vector_norm(self.v_torch.data, dim=-1, keepdim=True) + 1e-9)
            self.l_torch.data = self.l_torch.data/(LA.vector_norm(self.l_torch.data, dim=-1, keepdim=True) + 1e-9)
            self.refractive_index_torch.data = torch.clamp(self.refractive_index_torch.data, REFRACTIVE_INDEX_MIN, REFRACTIVE_INDEX_MAX)
            self.albedo_ratio_torch.data = torch.clamp(self.albedo_ratio_torch.data, min=0.0)
            self.k_s_torch.data = torch.clamp(self.k_s_torch.data, min=0.0)
            alpha_clamp = torch.clamp(self.params_torch[0].data, ALPHA_MIN, ALPHA_MAX)
            beta_clamp = torch.clamp(self.params_torch[1].data, BETA_MIN, BETA_MAX)
            self.params_torch.data = torch.stack([alpha_clamp, beta_clamp])
            self.kappa_torch.data = torch.clamp(self.kappa_torch, KAPPA_MIN, KAPPA_MAX)
            
            old_loss = loss.item()

    def numpy(self):
        n_norm = self.n_torch/LA.vector_norm(self.n_torch, dim=-1, keepdim=True)
        v_norm = self.v_torch/LA.vector_norm(self.v_torch, dim=-1, keepdim=True)
        l_norm = self.l_torch/LA.vector_norm(self.l_torch, dim=-1, keepdim=True)
    
        n = n_norm.detach().cpu().numpy()
        v = v_norm.detach().cpu().numpy()
        l = l_norm.detach().cpu().numpy()
        refractive_index = self.refractive_index_torch.detach().cpu().numpy()
        albedo_ratio = self.albedo_ratio_torch.detach().cpu().numpy()
        k_s = self.k_s_torch.detach().cpu().numpy()
        params = self.params_torch.detach().cpu().numpy()
        kappa = self.kappa_torch.detach().cpu().numpy()
        return n, v, l, refractive_index, albedo_ratio, k_s, params, kappa


def BRDFfitting_plane(slope_area_distribution, observed_stok, n, l, v, refractive_index_init, albedo_ratio_init, k_s_init, params_init, kappa_init, step_c, net, net_GAF, lr=0.01, iteration=10000, epsilon=1e-6, isPrint=True, joint_function=separable_GAF):
    opt = Optimizer_plane(slope_area_distribution, observed_stok, n, l, v, refractive_index_init, albedo_ratio_init, k_s_init, params_init, kappa_init, net, net_GAF, joint_function=joint_function)
    
    loss_fn = mse_loss_intensity()
    opt.refractive_index_torch.requires_grad_()
    opt.albedo_ratio_torch.requires_grad_()
    opt.albedo_ratio_torch.requires_grad_()
    opt.k_s_torch.requires_grad_()
    opt.params_torch.requires_grad_()
    opt.kappa_torch.requires_grad_()
    optimizer = optim.Adam([opt.refractive_index_torch, opt.albedo_ratio_torch, opt.k_s_torch, opt.params_torch, opt.kappa_torch], lr=lr)
    opt.optimize(optimizer, loss_fn, step_c, iteration=iteration, epsilon=epsilon, isPrint=isPrint)
    _,_,_, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est = opt.numpy()
    return refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est


# fitting to MERL BRDF dataset -------------------------------------------------
def rendering_MERL_torch(slope_area_distribution, net, net_GAF, n, v, l, refractive_index, rho, k_s, params, kappa, step_c, joint_function=separable_GAF):
    '''
    l is not list
    return BRDF
    '''
    device = n.device
    
    n_clamp = n/(LA.vector_norm(n, dim=-1, keepdim=True) + 1e-9)
    v_clamp = v/(LA.vector_norm(v, dim=-1, keepdim=True) + 1e-9)
    l_clamp = l/(LA.vector_norm(l, dim=-1, keepdim=True) + 1e-9)
    refractive_index_clamp = torch.clamp(refractive_index, REFRACTIVE_INDEX_MIN, REFRACTIVE_INDEX_MAX)
    rho_clamp = torch.clamp(rho, min=0.0)
    k_s_clamp = torch.clamp(k_s, min=0.0)
    alpha_clamp = torch.clamp(params[0], ALPHA_MIN, ALPHA_MAX)
    beta_clamp = torch.clamp(params[1], BETA_MIN, BETA_MAX)
    params_clamp = torch.stack([alpha_clamp, beta_clamp])
    kappa_clamp = torch.clamp(kappa, KAPPA_MIN, KAPPA_MAX)

    c = compute_c_isotropic(slope_area_distribution, params_clamp, step_c)

    # rendering
    GAF = GAFNN(n, l, v, params, net_GAF, joint_function=joint_function)

    n_v, v_v, l_v, R_c2v = transform_c2v(n, v, l)
    s_s = specular(slope_area_distribution, n_v, l_v, v_v, refractive_index, k_s, params, c).unsqueeze(-2)
    s_b = rho.unsqueeze(-1) * diffuse(net, n_v, l_v, v_v, refractive_index, 1.0, params, kappa).unsqueeze(-2)
    stok = GAF.unsqueeze(-1).unsqueeze(-1)*(s_s+s_b)

    # compute brdf from intensity
    cos_nl = torch.clamp(batchdot(n_clamp, l_clamp), 0.0, 1.0)
    brdf = stok[...,0]/(cos_nl + 1e-9).unsqueeze(-1)
    
    return brdf


class Renderer_BRDF:
    def __init__(self, slope_area_distribution, n, l, v, refractive_index, albedo_ratio, k_s, params, kappa):

        self.slope_area_distribution = slope_area_distribution
        self.n = n
        self.l = l
        self.v = v
        self.refractive_index = refractive_index
        self.albedo_ratio = albedo_ratio
        self.k_s = k_s
        self.rho = self.albedo_ratio * self.k_s
        self.params = params
        self.kappa = kappa

        self.num_brdf = n.shape[0]
        if len(self.v.shape) == 1:
            self.v = np.tile(self.v, (self.num_brdf, 1))

    def rendering(self, net, net_GAF, step_c, joint_function=separable_GAF):
        device = next(net.parameters()).device

        n_torch = torch.tensor(self.n, device=device, dtype=torch.float)
        v_torch = torch.tensor(self.v, device=device, dtype=torch.float)
        l_torch = torch.tensor(self.l, device=device, dtype=torch.float)
        refractive_index_torch = torch.tensor(self.refractive_index, device=device, dtype=torch.float)
        rho_torch = torch.tensor(self.rho, device=device, dtype=torch.float)
        k_s_torch = torch.tensor(self.k_s, device=device, dtype=torch.float)
        params_torch = torch.tensor(self.params, device=device, dtype=torch.float)
        kappa_torch = torch.tensor(self.kappa, device=device, dtype=torch.float)

        with torch.no_grad():
            model_brdf_torch = rendering_MERL_torch(self.slope_area_distribution, net, net_GAF, n_torch, v_torch, l_torch, refractive_index_torch, rho_torch, k_s_torch, params_torch, kappa_torch, step_c, joint_function=joint_function)
        
        # convert to numpy
        self.model_brdf = model_brdf_torch.detach().cpu().numpy()


# fitting with BRDF
class Optimizer_MERL:
    def __init__(self, slope_area_distribution, observed_BRDF, n, l, v, refractive_index, albedo_ratio, k_s, params, kappa, net, net_GAF, joint_function=separable_GAF):
       
        num_brdf = observed_BRDF.shape[0]
        if len(v.shape) == 1:
            v = np.tile(v, (num_brdf,1))
        
        device = next(net.parameters()).device
        self.net = net
        self.net_GAF = net_GAF
        self.joint_function = joint_function
        self.slope_area_distribution = slope_area_distribution

        # convert into tensor
        self.n_torch = torch.tensor(n, device=device, dtype=torch.float)
        self.v_torch = torch.tensor(v, device=device, dtype=torch.float)
        self.l_torch = torch.tensor(l, device=device, dtype=torch.float)
        self.refractive_index_torch = torch.tensor(refractive_index, device=device, dtype=torch.float)
        self.albedo_ratio_torch = torch.tensor(albedo_ratio, device=device, dtype=torch.float)
        self.k_s_torch = torch.tensor(k_s, device=device, dtype=torch.float)
        self.params_torch = torch.tensor(params, device=device, dtype=torch.float)
        self.kappa_torch = torch.tensor(kappa, device=device, dtype=torch.float)
        
        self.rho_torch = self.albedo_ratio_torch * self.k_s_torch
        self.observed_BRDF_torch = torch.tensor(observed_BRDF, device=device, dtype=torch.float)


    def optimize(self, optimizer, loss_fn, step_c, iteration, epsilon=1e-6, isPrint=True):
        old_loss = 0.0
        for i in range(iteration):
            optimizer.zero_grad()

            self.rho_torch = self.albedo_ratio_torch * self.k_s_torch
            brdf_torch = rendering_MERL_torch(self.slope_area_distribution, self.net, self.net_GAF, self.n_torch, self.v_torch, self.l_torch, self.refractive_index_torch, self.rho_torch, self.k_s_torch, self.params_torch, self.kappa_torch, step_c, joint_function=self.joint_function)
            
            loss = loss_fn(brdf_torch, self.observed_BRDF_torch)
            loss.backward()
            if isPrint:
                print(f"step: {i}, loss: {loss.item()}")
            if np.abs(old_loss-loss.item())/loss.item() < epsilon:
                print('complete')
                break
            optimizer.step()

            self.n_torch.data = self.n_torch.data/(LA.vector_norm(self.n_torch.data, dim=-1, keepdim=True) + 1e-9)
            self.v_torch.data = self.v_torch.data/(LA.vector_norm(self.v_torch.data, dim=-1, keepdim=True) + 1e-9)
            self.l_torch.data = self.l_torch.data/(LA.vector_norm(self.l_torch.data, dim=-1, keepdim=True) + 1e-9)
            self.refractive_index_torch.data = torch.clamp(self.refractive_index_torch.data, REFRACTIVE_INDEX_MIN, REFRACTIVE_INDEX_MAX)
            self.albedo_ratio_torch.data = torch.clamp(self.albedo_ratio_torch.data, min=0.0)
            self.k_s_torch.data = torch.clamp(self.k_s_torch.data, min=0.0)
            alpha_clamp = torch.clamp(self.params_torch[0].data, ALPHA_MIN, ALPHA_MAX)
            beta_clamp = torch.clamp(self.params_torch[1].data, BETA_MIN, BETA_MAX)
            self.params_torch.data = torch.stack([alpha_clamp, beta_clamp])
            self.kappa_torch.data = torch.clamp(self.kappa_torch.data, KAPPA_MIN, KAPPA_MAX)
            
            old_loss = loss.item()

    def numpy(self):
        n_norm = self.n_torch/LA.vector_norm(self.n_torch, dim=-1, keepdim=True)
        v_norm = self.v_torch/LA.vector_norm(self.v_torch, dim=-1, keepdim=True)
        l_norm = self.l_torch/LA.vector_norm(self.l_torch, dim=-1, keepdim=True)
    
        n = n_norm.detach().cpu().numpy()
        v = v_norm.detach().cpu().numpy()
        l = l_norm.detach().cpu().numpy()
        refractive_index = self.refractive_index_torch.detach().cpu().numpy()
        albedo_ratio = self.albedo_ratio_torch.detach().cpu().numpy()
        k_s = self.k_s_torch.detach().cpu().numpy()
        params = self.params_torch.detach().cpu().numpy()
        kappa = self.kappa_torch.detach().cpu().numpy()
        return n, v, l, refractive_index, albedo_ratio, k_s, params, kappa

def BRDFestimation_MERL(slope_area_distribution, observed_BRDF, n, l, v, refractive_index_init, albedo_ratio_init, k_s_init, params_init, kappa_init, step_c, net, net_GAF, lr=0.01, iteration=10000, loss_fn=nn.HuberLoss(), epsilon=1e-6, isPrint=True, joint_function=separable_GAF):
    opt = Optimizer_MERL(slope_area_distribution, observed_BRDF, n, l, v, refractive_index_init, albedo_ratio_init, k_s_init, params_init, kappa_init, net, net_GAF, joint_function=joint_function)
    
    opt.refractive_index_torch.requires_grad_()
    opt.albedo_ratio_torch.requires_grad_()
    opt.albedo_ratio_torch.requires_grad_()
    opt.k_s_torch.requires_grad_()
    opt.params_torch.requires_grad_()
    opt.kappa_torch.requires_grad_()
    optimizer = optim.Adam([opt.refractive_index_torch, opt.albedo_ratio_torch, opt.k_s_torch, opt.params_torch, opt.kappa_torch], lr=lr)
    opt.optimize(optimizer, loss_fn, step_c, iteration=iteration, epsilon=epsilon, isPrint=isPrint)
    _,_,_, refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est = opt.numpy()
    return refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est

def MERL_initial_estimation(slope_area_distribution, observed_BRDF, n, l, v, refractive_index_sampled, rho_sampled, k_s_sampled, params_sampled, kappa, step_c, net, net_GAF, loss=None, joint_function=separable_GAF):
    device = next(net.parameters()).device
    len1 = len(refractive_index_sampled)
    len2 = len(params_sampled)
    len3 = len(rho_sampled)
    len4 = len(k_s_sampled)
    num_brdf = len(observed_BRDF)
    color_num = observed_BRDF.shape[-1] ###
    observed_BRDF_torch = torch.tensor(observed_BRDF, device=device, dtype=torch.float)

    with torch.no_grad():
        n_torch = torch.tensor(n, device=device, dtype=torch.float, requires_grad=False)
        l_torch = torch.tensor(l, device=device, dtype=torch.float, requires_grad=False)
        v_torch = torch.tensor(v, device=device, dtype=torch.float, requires_grad=False)
        refractive_index_sampled_torch = torch.tensor(refractive_index_sampled, device=device, dtype=torch.float, requires_grad=False)
        rho_sampled_torch = torch.tensor(rho_sampled, device=device, dtype=torch.float, requires_grad=False)
        k_s_sampled_torch = torch.tensor(k_s_sampled, device=device, dtype=torch.float, requires_grad=False)
        params_sampled_torch = torch.tensor(params_sampled, device=device, dtype=torch.float, requires_grad=False)
        kappa_torch = torch.tensor(kappa, device=device, dtype=torch.float, requires_grad=False)

        c_dist_sampled = torch.zeros((len2, ))
        for i, params in enumerate(params_sampled_torch):
            c_dist_sampled[i] = compute_c_isotropic(slope_area_distribution, params, step_c)

        loss_list = torch.zeros((len1, len2, len4), device=device, dtype=torch.float) ###
        rho_list = np.zeros((len1, len2, len4, color_num)) ###
        # compute loss with sampled parameters
        for idx1, refractive_index in enumerate(refractive_index_sampled_torch):
            print('idx1:', idx1)
            for idx2, params in enumerate(params_sampled_torch):
                print('idx2:', idx2)
                c = c_dist_sampled[idx2]
                GAF = GAFNN(n_torch, l_torch, v_torch, params, net_GAF, joint_function=joint_function)

                n_v, v_v, l_v, R_c2v = transform_c2v(n_torch, v_torch, l_torch)
                s_b_list = diffuse(net, n_v, l_v, v_v, refractive_index, 1.0, params, kappa_torch)
                s_s_list = specular(slope_area_distribution, n_v, l_v, v_v, refractive_index, 1.0, params, c)
                I_b_list = GAF*s_b_list[...,0]
                I_s_list = GAF*s_s_list[...,0]
                cos_nl = torch.clamp(torch.sum(n_torch*l_torch, dim=-1), 1e-18, 1.0)

                for idx4, k_s in enumerate(k_s_sampled_torch):
                    # minimize the loss for each color.
                    # rho is independent for each channel.
                    loss_temp = 0.0
                    for idx_col in range(color_num):
                        loss_list_temp = torch.zeros((len3, ), device=device, dtype=torch.float)
                        for idx3, rho in enumerate(rho_sampled_torch):
                            model_brdf = (rho*I_b_list + k_s * I_s_list)/cos_nl
                            if loss is None:
                                loss_list_temp[idx3] = torch.sum((observed_BRDF_torch[...,idx_col] - model_brdf)**2)/num_brdf
                            else:
                                loss_list_temp[idx3] = loss(model_brdf, observed_BRDF_torch[...,idx_col])
                        idx3_min = torch.argmin(loss_list_temp).cpu().detach().numpy()
                        rho_list[idx1, idx2, idx4, idx_col] = rho_sampled[idx3_min]
                        loss_temp += loss_list_temp[idx3_min]
                    loss_list[idx1,idx2,idx4] = loss_temp/color_num

        # return the initial values
        idx_min = torch.argmin(loss_list).cpu().detach().numpy()
        temp, idx4_min = divmod(idx_min, len4)
        temp, idx2_min = divmod(temp, len2)
        temp, idx1_min = divmod(temp, len1)

        refractive_index_init = refractive_index_sampled[idx1_min]
        params_init = params_sampled[idx2_min]
        rho_init = rho_list[idx1_min, idx2_min, idx4_min]
        k_s_init = k_s_sampled[idx4_min]
    return refractive_index_init, params_init, rho_init, k_s_init


def MERL_robust_optimization(slope_area_distribution, observed_BRDF, n, l, v, refractive_index_sampled, rho_sampled, k_s_sampled, params_sampled, kappa_init, step_c, net, net_GAF, lr=0.003, iteration=10000, loss_fn = nn.HuberLoss(), epsilon=1e-5, isPrint=True, loss_init=None, joint_function=separable_GAF):
    device = next(net.parameters()).device
    shape = observed_BRDF.shape[0:-1]
    if len(v.shape) == 1:
        v = np.tile(v, shape + (1,))

    refractive_index_init, params_init, rho_init, k_s_init = MERL_initial_estimation(slope_area_distribution, observed_BRDF, n, l, v, refractive_index_sampled, rho_sampled, k_s_sampled, params_sampled, kappa_init, step_c, net, net_GAF, loss=loss_init, joint_function=joint_function)
    print('initial value')
    print(f'refractive_index: {refractive_index_init}')
    print(f'rho: {rho_init}')
    print(f'k_s: {k_s_init}')
    print(f'params: {params_init}')
    albedo_ratio_init = rho_init/k_s_init

    refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est = BRDFestimation_MERL(slope_area_distribution, observed_BRDF, n, l, v, refractive_index_init, albedo_ratio_init, k_s_init, params_init, kappa_init, step_c, net, net_GAF, lr=lr, iteration=iteration, loss_fn=loss_fn, epsilon=epsilon, isPrint=isPrint, joint_function=joint_function)

    return refractive_index_est, albedo_ratio_est, k_s_est, params_est, kappa_est
