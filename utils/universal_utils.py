"""
This file contains some codes derived from the open-source project.
"""
import torch.nn as nn
import torch.nn.functional as F
from time import time
import torch
import numpy as np
from model.smpl.pytorch.smpl import SMPL
import utils.pytorch3d_transforms as trans3d

def get_stand_param(batch_size=1,is_torch=True,device="cuda"):
    t_shape = np.zeros((batch_size, 10), dtype=np.float32)
    t_trans = np.zeros((batch_size, 3), dtype=np.float32)

    pose_param = np.zeros((batch_size, 24, 3), dtype=np.float32)
    divdie_num = 3
    pose_param[:,16,2] = -torch.pi / divdie_num
    pose_param[:,17,2] = torch.pi / divdie_num
    t_pose = np.reshape(pose_param, (batch_size, 72))

    if(is_torch):
        return torch.tensor(t_pose).to(device), torch.tensor(t_shape).to(device), torch.tensor(t_trans).to(device)
    return t_pose, t_shape,t_trans

def get_stand_pose_smpl(gender_type='neutral',is_show=False,is_face=False):
    smpl = SMPL(gender=gender_type)

    t_pose, t_shape,t_trans=get_stand_param()

    d, J = get_smpl_result(t_pose, t_shape, t_trans, smpl)

    if(is_face):
        return d,J,smpl.f
    return d,J


def get_smpl_result(pose_param,shape_param,trans_param,SMPL,is_havemodel=True):
    if(type(pose_param)!=torch.Tensor):
        pose_param=torch.cuda.FloatTensor(pose_param)
        shape_param=torch.cuda.FloatTensor(shape_param)
        trans_param=torch.cuda.FloatTensor(trans_param)

    d ,J= SMPL(pose_param, shape_param,trans_param)

    return d,J


# 兼容 2/3 维 tensor/ numpy
def get_corr_idx(s1, s2, dim=-1):
    is_numpy = isinstance(s1, np.ndarray) or isinstance(s2, np.ndarray)
    if(is_numpy):
        s1=torch.tensor(s1)
        s2=torch.tensor(s2)
    distance_matrix = torch.cdist(s1, s2)

    corr_idx=torch.argmin(distance_matrix, dim=dim)
    if(is_numpy):
        corr_idx=corr_idx.cpu().detach().numpy()
    return corr_idx

def my_chamfer_distance(x, y, first_dim=0, scale=0.5, is_return_dist=False):
    distance_matrix = torch.cdist(x, y)
    #distance_matrix*=50
    # print(distance_matrix[:10,:10])
    av_dist1 = torch.min(distance_matrix, first_dim+1)[0]
    av_dist2 = torch.min(distance_matrix, first_dim)[0]

    if (is_return_dist):
        return av_dist1, av_dist2
    return scale * torch.mean(av_dist1) + (1 - scale) * torch.mean( av_dist2)

def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """

    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))

    return res.reshape(*raw_size, -1)


def surreal_coordinate_transform(data):
    k = data[..., 0].clone()
    # 不应该乘-1
    data[..., 0] = data[..., 2]
    data[..., 2] = k
    data[...,1]=data[...,1]*-1
    return data

def get_smpl_result_no_humanmodel(pose_param,shape_param,trans_param,gender="male",model_root='model/smpl/pytorch/models/',return_pelvis=False):
    smpl = SMPL(model_root=model_root,gender=gender).cuda()
    if(type(pose_param)!=torch.Tensor):
        pose_param=torch.cuda.FloatTensor(pose_param)
        shape_param=torch.cuda.FloatTensor(shape_param)
        trans_param=torch.cuda.FloatTensor(trans_param)
    if(len(pose_param.shape)==1):
        pose_param=pose_param.unsqueeze(0)
        shape_param=shape_param.unsqueeze(0)
        trans_param=trans_param.unsqueeze(0)

    if(return_pelvis):
        d ,J,pelvis= smpl(pose_param, shape_param,trans_param,return_pelvis=True)
        return d,J,pelvis

    d ,J= smpl(pose_param, shape_param,trans_param)


    return d,J

def get_possible_smpl(batch_size,expand_num,pose_param=None,shape_param=None,trans_param=None,use_fix=False):

    set_axis = torch.tensor([[0, 0, 0],[0, np.pi, 0],[0,- np.pi/2, 0],[0, np.pi/2, 0],[-np.pi/2, 0, 0],[np.pi/2, 0, 0],[1, 1, 0],[1, 0, 1]]) #
    set_axis=set_axis[:expand_num]
    expand_num=set_axis.shape[0]
    if(use_fix):
        pose_param,shape_param,trans_param=get_stand_param(batch_size*expand_num)

    pose_param=pose_param.reshape(batch_size,expand_num,-1)

    rotvec = trans3d.matrix_to_axis_angle(trans3d.euler_angles_to_matrix(set_axis,"XYZ"))
    pose_param[:,:,:3]=rotvec
    pose_param=pose_param.reshape(-1,72)
    return pose_param,shape_param,trans_param



def hyperextend_jud(pred_kp,is_star=True):

    if(is_star):
        LEFT_KNEE=4
        RIGHT_KNEE=5
        LEFT_HIP=1
        RIGHT_HIP=2
        LEFT_ANKLE=7
        RIGHT_ANKLE=8
        LEFT_COLLAR = 13
        RIGHT_COLLAR = 14
    else:
        pass


    left_hip_vector = pred_kp[:, LEFT_KNEE] - pred_kp[:, LEFT_HIP]
    left_ankle_vector = pred_kp[:, LEFT_KNEE] - pred_kp[:, LEFT_ANKLE]

    right_hip_vector = pred_kp[:, RIGHT_KNEE] - pred_kp[:, RIGHT_HIP]
    right_ankle_vector = pred_kp[:, RIGHT_KNEE] - pred_kp[:, RIGHT_ANKLE]

    left_leg_vector = left_ankle_vector + left_hip_vector
    right_leg_vector = right_ankle_vector + right_hip_vector


    leg_vector = left_leg_vector + right_leg_vector

    # leg_vector = leg_vector
    leg_norm = torch.norm(leg_vector, dim=1, p=2)

    shoulder_vector = pred_kp[:, RIGHT_COLLAR] - pred_kp[:, LEFT_COLLAR]
    down_vector = torch.tensor([0, -1, 0]).cuda().to(torch.float32)
    approximate_direction = torch.cross(shoulder_vector, down_vector.unsqueeze(0), dim=1)

    result=torch.cosine_similarity(leg_vector[:, :2], approximate_direction[:,:2], dim=1)

    return result



def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    return torch.sum((src[:, :, None] - dst[:, None]) ** 2, dim=-1)


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S, [K]]
    Return:
        new_points:, indexed points data, [B, S, [K], C]
    """

    raw_size = idx.size()
    idx = idx.reshape(raw_size[0], -1)
    res = torch.gather(points, 1, idx[..., None].expand(-1, -1, points.size(-1)))

    return res.reshape(*raw_size, -1)


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        distance = torch.min(distance, dist)
        farthest = torch.max(distance, -1)[1]

    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False, knn=False):
    """
    Input:
        npoint:
        radius:
        nsample:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, npoint, nsample, 3]
        new_points: sampled points data, [B, npoint, nsample, 3+D]
    """
    B, N, C = xyz.shape
    S = npoint

    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint]
    torch.cuda.empty_cache()

    new_xyz = index_points(xyz, fps_idx)

    torch.cuda.empty_cache()
    if knn:
        dists = square_distance(new_xyz, xyz)  # B x npoint x N
        idx = dists.argsort()[:, :, :nsample]  # B x npoint x K
    else:
        idx = query_ball_point(radius, nsample, xyz, new_xyz)
    torch.cuda.empty_cache()
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    torch.cuda.empty_cache()
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)
    torch.cuda.empty_cache()


    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm

    if returnfps:
        return new_xyz, new_points, fps_idx
        # return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    """
    Input:
        xyz: input points position data, [B, N, 3]
        points: input points data, [B, N, D]
    Return:
        new_xyz: sampled points position data, [B, 1, 3]
        new_points: sampled points data, [B, 1, N, 3+D]
    """
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all, knn=False):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.knn = knn
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C]
            points: input points data, [B, N, C]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points,fp_idx = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points, returnfps=True,knn=self.knn)

        # new_xyz: sampled points position data, [B, npoint, C]
        # new_points: sampled points data, [B, npoint, nsample, C+D]
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0].transpose(1, 2)

        return new_xyz, new_points,fp_idx


def coordinate_transform(data):
    # 复制原始Y轴的值
    original_y = data[..., 1].clone()

    # 交换Y和Z轴，并对新Z轴（原Y轴）取反
    data[..., 1] = data[..., 2]  # Y <- Z
    data[..., 2] = -original_y  # Z <- -Y

    # X轴保持不变
    return data