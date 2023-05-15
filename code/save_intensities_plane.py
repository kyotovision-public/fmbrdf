import numpy as np
import cv2
import os
import sys
import glob
import argparse

sys.path.append(os.path.join(os.path.dirname('__file__'), '..'))
import utils.pimg_smooth_mono as ps
import utils.plane_board_utils as pu

def get_option():
    argparser = argparse.ArgumentParser(description='Preprocess')
    argparser.add_argument('OBJECT_NAME')
    argparser.add_argument('ROTATION', type=float, help='interval of camera rotation')
    argparser.add_argument('ROWS', type=int, help='the number of rows of ChArUco borad')
    argparser.add_argument('COLUMNS', type=int, help='the number of columns of ChArUco borad')
    argparser.add_argument('SQUARELENGTH', type=float, help='checker size')
    argparser.add_argument('MARKERLENGTH', type=float, help='marker size')
    argparser.add_argument('CORNER_IDS', type=int, nargs=4, help='upper left, upper right, lower right, lower left')
    argparser.add_argument('-s','--start', type=int, help='offset of start', default=0)
    argparser.add_argument('-e','--end', type=int, help='offset of end', default=0)
    return argparser.parse_args()
args = get_option()
obj_name = args.OBJECT_NAME
rot_cam = args.ROTATION
rows = args.ROWS
cols = args.COLUMNS
squareLength = args.SQUARELENGTH
markerLength = args.MARKERLENGTH
corner_ids = args.CORNER_IDS
start = args.start
end = args.end

img_gain = 4.0 # gain for marker detection
offset_x = 100
offset_y = 100
rotate_mask_offset_x = 10
dict_aruco = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)

# folder file
DATA_PATH = '../data/'
IMAGE_PATH = DATA_PATH+obj_name+'/image/img_ud/'
distortion_file = DATA_PATH+obj_name+'/calib/distortionL.npy'
intrinsic_file = DATA_PATH+obj_name+'/calib/IntrinsicL.npy'
light_file = DATA_PATH+obj_name+'/calib/light.npz'
savefile_npz = DATA_PATH + obj_name + '/data.npz'

# load files
distortion = np.load(distortion_file)
intrinsic = np.load(intrinsic_file)

# sort TASK
img_files = sorted(glob.glob(IMAGE_PATH+'*.png'))
l_num = len(img_files)

# obtain intensity of plane with mesogeometry
theta_nv = np.zeros((l_num))
intensity_list = np.zeros((l_num))
stokes_list = np.zeros((l_num, 4))
normal_list = np.zeros((l_num, 3))
pose_list = np.zeros((l_num, 3, 3))
valid_mask = np.zeros((l_num), dtype=bool)
previous_mask = None
previous_pose = None
for i, img_file in enumerate(img_files):
    scene = os.path.splitext(os.path.basename(img_file))[0]
    print(scene)
    if i < start or i >= (l_num - end):
        continue

    pimg = cv2.imread(img_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE).astype(np.float32) /(256*256-1.0)
    img = np.clip(pimg * 256 * img_gain, 0, 255).astype(np.uint8)
    board = cv2.aruco.CharucoBoard_create(cols,rows, squareLength, markerLength, dict_aruco)
    parameters = cv2.aruco.DetectorParameters_create()

    # detect marker
    h, w = pimg.shape
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, dict_aruco, parameters=parameters)
    mask_full = pu.make_mask(corners, ids, corner_ids, h, w, offset_x=offset_x, offset_y=offset_y)
    pose = None # pose of board
    if len(corners) != 0:
        # detect checkerboard
        retval_b, charucoCorners_b, charucoIds_b = cv2.aruco.interpolateCornersCharuco(corners, ids, img, board)
        if (charucoCorners_b is not None):
            # estimate pose of checkerboard
            valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners_b, charucoIds_b, board, intrinsic, distortion, None, None)
            if valid:
                pose = cv2.Rodrigues(rvec)[0]
                print('pose', pose)

    
    # if checkerboard cannot be detected, rotate previous mask and pose
    pred_mask = None
    pred_pose = None
    if (previous_mask is not None) and (previous_pose is not None):
        pred_mask, pred_pose = pu.rotate_mask_pose(previous_mask, previous_pose, rot_cam, rotate_mask_offset_x)
    if (mask_full is None) and (pred_mask is not None):
        mask_full = pred_mask.copy()
    if (pose is None) and (pred_pose is not None):
        pose = pred_pose.copy()

    # save
    if (mask_full is not None) and (pose is not None):
        normal = -pose[:,2]
        mask = mask_full[::2,::2]

        pbayer_ = ps.PBayer(pimg)
        I_DC = pbayer_.I_dc

        # compute average intensity within mask
        if np.sum(mask) != 0:
            stokes = pbayer_.stokes
            stokes_mean = np.mean(stokes[mask], axis=0)        
            theta_nv[i] = np.arccos(-normal[2])*np.sign(-normal[0])
            print('pose', pose)
            print('normal', normal)
            print('theta_nv: ', theta_nv[i]*180/np.pi)
            intensity_list[i] = stokes_mean[0]
            stokes_list[i] = stokes_mean.copy()
            pose_list[i] = pose.copy()
            normal_list[i] = normal.copy()

            previous_mask = mask_full.copy()
            previous_pose = pose.copy()
            valid_mask[i] = True



previous_mask = None
previous_pose = None
# reversely rotate mask
for i_inv in range(l_num):
    i = l_num - i_inv - 1
    img_file = img_files[i]
    scene = os.path.splitext(os.path.basename(img_file))[0]
    print(scene)
    if i < start or i >= (l_num - end):
        continue

    pimg = cv2.imread(img_file, cv2.IMREAD_ANYDEPTH | cv2.IMREAD_GRAYSCALE).astype(np.float32) /(256*256-1.0)
    img = np.clip(pimg * 256 * img_gain, 0, 255).astype(np.uint8)
    board = cv2.aruco.CharucoBoard_create(cols,rows, squareLength, markerLength, dict_aruco)
    parameters = cv2.aruco.DetectorParameters_create()

    # detect marker
    h, w = pimg.shape
    corners, ids, rejectedImgPoints = cv2.aruco.detectMarkers(img, dict_aruco, parameters=parameters)
    mask_full = pu.make_mask(corners, ids,corner_ids, h, w, offset_x=offset_x, offset_y=offset_y)
    pose = None
    if len(corners) != 0:
        # detect checkerboard
        retval_b, charucoCorners_b, charucoIds_b = cv2.aruco.interpolateCornersCharuco(corners, ids, img, board)
        if (charucoCorners_b is not None):
            # estimate pose of checkerboard
            valid, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(charucoCorners_b, charucoIds_b, board, intrinsic, distortion, None, None)
            if valid:
                pose = cv2.Rodrigues(rvec)[0]
                print('pose', pose)
    

    # if checkerboard cannot be detected, rotate previous mask and pose
    pred_mask = None
    pred_pose = None
    if (previous_mask is not None) and (previous_pose is not None):
        pred_mask, pred_pose = pu.rotate_mask_pose(previous_mask, previous_pose, -rot_cam, rotate_mask_offset_x)
    if (mask_full is None) and (pred_mask is not None):
        mask_full = pred_mask.copy()
    if (pose is None) and (pred_pose is not None):
        pose = pred_pose.copy()

    # save
    if (mask_full is not None) and (pose is not None):
        # ignore those already computed
        if (not valid_mask[i]):
            normal = -pose[:,2]
            mask = mask_full[::2,::2]

            pbayer_ = ps.PBayer(pimg)
            I_DC = pbayer_.I_dc

            # compute average intensity within mask
            if np.sum(mask) != 0:
                stokes = pbayer_.stokes
                stokes_mean = np.mean(stokes[mask], axis=0)        
                theta_nv[i] = np.arccos(-normal[2])*np.sign(-normal[0])
                print('normal', normal)
                print('theta_nv: ', theta_nv[i]*180/np.pi)
                intensity_list[i] = stokes_mean[0]
                stokes_list[i] = stokes_mean.copy()
                pose_list[i] = pose.copy()
                normal_list[i] = normal.copy()
                valid_mask[i] = True
        
        previous_mask = mask_full.copy()
        previous_pose = pose.copy()

# obtain light source direction
npz_comp = np.load(light_file)
l = npz_comp['light']
light_index = npz_comp['light_index']
pose_ref = pose_list[light_index]
l_list = np.zeros((l_num,3))
for i, pose in enumerate(pose_list):
    l_list[i] = pose @ pose_ref.T @ l

# save
np.savez_compressed(savefile_npz, intensity_list=intensity_list, stokes_list=stokes_list, pose_list=pose_list, light_list=l_list, normal_list=normal_list, valid_mask=valid_mask)
