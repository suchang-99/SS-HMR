import time

from torch.utils.data import Dataset

import pickle
import pathlib
import open3d as o3d
import numpy as np
import json
import cv2
import os
import sys
import torch

from utils.pytorch3d_transforms import matrix_to_axis_angle,axis_angle_to_matrix
import random



def transform_points(points, R, T):
    """ Transform 3D points from world coordinate system to camera coordinate system
    Args:
        points (np.ndarray): 3D points in world coordinate system of shape (N, 3).
        R (np.ndarray): world2cam rotation matrix of shape (3, 3).
        T (np.ndarray): world2cam translation vector of shape (3,).

    Returns:
        transformed_points (np.ndarray): 3D points in camera coordiante system of shape (N, 2).
    """
    N = points.shape[0]

    # compute world to camera transformation
    T_world2cam = np.eye(4)
    T_world2cam[:3, :3] = R
    T_world2cam[:3, 3] = T

    # convert 3D points to homogeneous coordinates
    points_3D = points.T  # (3, N)
    points_homo = np.vstack([points_3D, np.ones((1, N))])  # (4, N)

    # transform points to the camera frame
    transformed_points = T_world2cam @ points_homo  # (4, N)
    transformed_points = transformed_points[:3, :]  # (3, N)
    transformed_points = transformed_points.T  # (N, 3)

    return transformed_points

def transform_coordinates(R1, T1, R2, T2, points_1):
    """
    对齐两个相机坐标系中的点。

    参数:
    R1, T1: 第一个相机的旋转矩阵和平移向量。
    R2, T2: 第二个相机的旋转矩阵和平移向量。
    points_1: 第一个相机坐标系中的点的坐标列表，每个点为 [x, y, z]。

    返回:
    points_2: 对齐后在第1个相机坐标系中的点的坐标列表。
    """
    # 计算相对旋转矩阵和平移向量
    R_relative = np.dot(R2.T, R1)
    T_relative = np.dot(R2.T, (T1 - T2)).reshape(3, 1)

    # 应用变换
    points_2 = np.dot(R_relative, points_1.T) + T_relative

    # 转置回来以匹配原始点云的尺寸
    points_2 = points_2.T

    return points_2




def compute_transform_from_camera_params(camera_params_src, camera_params_dst):
    # Compute color camera transformation
    T_world2src = np.eye(4)
    T_world2src[:3, :3] = camera_params_src['R']
    T_world2src[:3, 3] = camera_params_src['T']

    # Compute depth camera transformation
    T_world2dst = np.eye(4)
    T_world2dst[:3, :3] = camera_params_dst['R']
    T_world2dst[:3, 3] = camera_params_dst['T']

    # Compute depth2color transformation
    T_src2dst = T_world2dst @ np.linalg.inv(T_world2src)

    return T_src2dst

def get_all_smpl_data(smpl_params_path):
    smpl_params = np.load(smpl_params_path)
    global_orient = smpl_params['global_orient']
    body_pose = smpl_params['body_pose']
    betas = smpl_params['betas']
    transl = smpl_params['transl']
    return global_orient, body_pose, betas, transl

def transform_ori_params(global_orient,R, T,rR,rT):
    # Transform SMPL parameters to the new camera coordinate system
    global_orient = global_orient

    R_relative = np.dot(rR.T, R)
    raw_R=axis_angle_to_matrix(torch.tensor(global_orient))

    result=torch.bmm(torch.tensor(R_relative,dtype=torch.float32).unsqueeze(0).expand(raw_R.shape[0], -1, -1),raw_R)

    axis_angle = matrix_to_axis_angle(result)

    global_orient = torch.tensor(axis_angle)

    return global_orient

class BEHAVEDataLoader(Dataset):

    def __init__(self, cfg, mode):

        self.cfg = cfg
        self.mode = mode

        if(mode=="train"):
            self.data_list=np.load("data/BEHAVE/processed/train_data.npy",allow_pickle=True)
        else:
            self.data_list=np.load("data/BEHAVE/processed/test_data.npy",allow_pickle=True)
        tmp_data_list=[]
        for tmp_i in range(0, len(self.data_list)):
            self.data_list[tmp_i]["data_dir"] = self.data_list[tmp_i]["data_dir"].replace("../../SHMR/","")
            self.data_list[tmp_i]["data_dir"] = self.data_list[tmp_i]["data_dir"].replace("\\","/")
            with open( self.data_list[tmp_i]["data_dir"], 'rb') as f:
                pc_data= pickle.load(f, encoding='bytes')
            if(pc_data.shape[1]==1024):
                tmp_data_list.append(self.data_list[tmp_i])
            else:
                print(self.data_list[tmp_i]["data_dir"])
            
        self.data_list=tmp_data_list
        self.len=len(self.data_list)



    # def self_orient_correct(self):

    def ini_pesudo_label(self):
        for tmp_i in range(0, self.len):
            self.data_list[tmp_i]["ref_direction"] = torch.zeros(3)
            self.data_list[tmp_i]["pseudo_kp"] = torch.zeros((21, 3))

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        single_data = self.data_list[index]
        data_file = single_data['data_dir']
        with open(data_file, 'rb') as f:
            pc_data= pickle.load(f, encoding='bytes')
        single_data['point_cloud'] = pc_data
        single_data['index']=index

        return single_data

    def process_depth_mask(self,mask,last_mask):
        confuse_mask=cv2.bitwise_and(mask,255-last_mask)//255
        sure_mask=cv2.bitwise_and(mask,last_mask)//255
        sure_mask= cv2.dilate(sure_mask, np.ones((6, 6), np.uint8), iterations=1)

        filter_mask=cv2.bitwise_and(confuse_mask,1-sure_mask)

        new_mask=mask-filter_mask*255
        return new_mask


