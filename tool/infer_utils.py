import os
import numpy as np
import cv2
from skimage import morphology
import torch

def cam_npy_to_cam_dict(cam_np, label):
    cam_dict = {}
    idxs = np.where(label==1)[0]
    for idx in idxs:
        cam_dict[idx] = cam_np[idx]
    return cam_dict

def dict2npy(cam_dict, gt_label, orig_img):
    if torch.is_tensor(gt_label):
        gt_label_np = gt_label.detach().cpu().numpy()
    else:
        gt_label_np = np.array(gt_label)

    gt_cat = np.where(gt_label_np == 1)[0]

    H, W = orig_img.shape[:2]
    bg_score = [gen_bg_mask(orig_img)]
    cam_npy = np.zeros((4, H, W), dtype=np.float32)

    if len(gt_cat) == 0:
        return cam_npy, bg_score

    for gt in gt_cat:
        if gt in cam_dict:
            cam_npy[gt] = cam_dict[gt]
    return cam_npy, bg_score


def gen_bg_mask(orig_img):
    img_array = np.array(orig_img).astype(np.uint8)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    binary = np.uint8(binary)    
    dst = morphology.remove_small_objects(binary==255,min_size=50,connectivity=1)
    bg_mask = np.zeros(orig_img.shape[:2])
    bg_mask[dst==True]=1.000001
    return bg_mask

def cam_npy_to_label_map(cam_npy):
    seg_map = cam_npy.transpose(1,2,0)
    seg_map = np.asarray(np.argmax(seg_map, axis=2), dtype=int)
    return seg_map

def cam_npy_to_cam_dict(cam_npy, label):
    cam_dict = {}
    for i in range(len(label)):
        if label[i] > 1e-5:
            cam_dict[i] = cam_npy[i]
    return cam_dict





















