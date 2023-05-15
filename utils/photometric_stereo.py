import numpy as np

# mono
def PS(intensity, mask_list, lights):
    image_num = intensity.shape[0]
    h = intensity.shape[1]
    w = intensity.shape[2]
    pixel_num = h*w

    nmap = np.zeros((h, w, 3))
    I = intensity.reshape(image_num, pixel_num) # shape: (number of images, number of pixels)

    # compute normal for each pixel
    mask_list_ = mask_list.reshape((image_num, pixel_num))
    n_tilde = np.ones((pixel_num, 3))*np.nan
    for i in range(pixel_num):
        mask_pixel = mask_list_[:,i]
        I_ = I[:,i][mask_pixel]
        lights_ = lights[mask_pixel]
        T_ = np.linalg.pinv(lights_)
        n_tilde[i] = np.dot(T_, I_)
    
    n = n_tilde/np.linalg.norm(n_tilde,axis=-1,keepdims=True)
    nmap = n.reshape((h,w,3))

    return nmap
