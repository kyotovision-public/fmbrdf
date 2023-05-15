import numpy as np

def make_mask(corners, ids, corner_ids, h, w, offset_x=150, offset_y=100):
    '''
    Obtain the region framed by the ChArUco board.
    corner_ids: ids of ArUco markers at (upper left, upper right, lower right, lower downleft)
    '''
    shape = (h,w)
    if ids is None:
        return None
    ids_copy = ids.copy()[:,0]
    corners_copy = np.array(corners).copy()[:,0,:,:] # shape: (num, 1, 4, 2) -> (num, 4, 2)

    markers = [] # [upleft_marker, upright_marker, downright_marker, downleft_marker]

    for i in range(4):
        temp_index = np.where(ids_copy == corner_ids[i])[0]
        if len(temp_index) == 0:
            return None
        temp_index = temp_index[0]
        markers.append(corners_copy[temp_index].copy())


    upleft = markers[0][3]
    upright = markers[1][2]
    downright = markers[2][1]
    downleft = markers[3][0]

    upleft[0] += offset_x
    upleft[1] += +offset_y
    upright[0] -= offset_x
    upright[1] += +offset_y
    downright[0] -= offset_x
    downright[1] -= +offset_y
    downleft[0] += offset_x
    downleft[1] -= +offset_y

    left = np.max([upleft[0], downleft[0]])
    up = np.max([upleft[1], upright[1]])
    right = np.min([upright[0], downright[0]])
    down = np.min([downleft[1], downright[1]])

    x = np.arange(w)
    y = np.arange(h)
    xx,yy = np.meshgrid(x,y)
    mask = np.ones(shape, dtype=bool)
    mask = mask & (xx > left)
    mask = mask & (yy > up)
    mask = mask & (xx < right)
    mask = mask & (yy < down)
    return mask

def rotate_mask_pose(previous_mask, previous_pose, rot_cam, offset_x = 0):
    '''
    Compute mask and camera pose from the previous them if mask is not detected.
    rot_cam: rotation angle of viewing direction. positive:z->x
    '''
    shape = previous_mask.shape
    h,w = previous_mask.shape

    # compute normal
    R = np.zeros((3,3)) # Transfomation from previous CS into current CS
    R[0,0] = np.cos(rot_cam)
    R[0,2] = -np.sin(rot_cam)
    R[1,1] = 1
    R[2,0] = np.sin(rot_cam)
    R[2,2] = np.cos(rot_cam)
    current_pose = R @ previous_pose
    current_normal = -current_pose[:,2]
    current_cos_nv = -current_normal[2]

    if np.sum(previous_mask) == 0:
        return previous_mask, current_pose

    x = np.arange(w)
    y = np.arange(h)
    xx, yy = np.meshgrid(x,y)
    center_x = np.mean(xx[previous_mask]) # center of mask

    # compute the width of rotated mask
    previous_normal = -previous_pose[:,2]
    previous_cos_nv = -previous_normal[2]
    previous_left = np.min(xx[previous_mask])
    previous_right = np.max(xx[previous_mask])
    rect_w = (previous_right - previous_left)/previous_cos_nv # the width of rectangle when viewd from the front
    current_rect_w = rect_w*current_cos_nv
    if current_rect_w < 0:
        current_rect_w = 0
    up = np.min(yy[previous_mask])
    down = np.max(yy[previous_mask])
    current_left = center_x - current_rect_w/2 + offset_x/2
    current_right = center_x + current_rect_w/2 - offset_x/2

    # make mask
    current_mask = np.ones(shape, dtype=bool)
    current_mask = current_mask & (xx > current_left)
    current_mask = current_mask & (yy > up)
    current_mask = current_mask & (xx < current_right)
    current_mask = current_mask & (yy < down)
    return current_mask, current_pose

