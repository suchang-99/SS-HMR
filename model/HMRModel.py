"""
Modified Work Based on DPC
===========================================================

**Original Project**: DPC: Unsupervised Deep Point Correspondence via Cross and Self Construction
**Source URL**: https://github.com/dvirginz/DPC

Written by Chang Su
"""

import numpy as np
import torch
import torch.nn as nn
from model.PointTransformerV3 import PointTransformerV3
import utils.universal_utils as uni_utils
import utils.pytorch3d_transforms as trans3d
from model.smpl.pytorch.smpl import SMPL
import torch.nn.functional as F

class TransitionDown(nn.Module):
    def __init__(self, k, nneighbor, channels):
        super().__init__()
        self.sa = uni_utils.PointNetSetAbstraction(k, 0, nneighbor, channels[0], channels[1:], group_all=False, knn=True)

    def forward(self, xyz, points):
        return self.sa(xyz, points)

def measure_similarity(similarity_init, source_encoded, target_encoded):
    """
    Measure the similarity between two batched matrices vector by vector

    Args:
        similarity_init : The method to calculate similarity with(e.g cosine)
        source_encoded (BxNxF Tensor): The input 1 matrix
        target_encoded (BxNxF Tensor): The input 2 matrix
    """
    "multiplication", "cosine", "difference"
    if similarity_init == "cosine":
        a_norm = source_encoded / source_encoded.norm(dim=-1)[:, :, None]
        b_norm = target_encoded / target_encoded.norm(dim=-1)[:, :, None]
        return torch.bmm(a_norm, b_norm.transpose(1, 2))
    if similarity_init == "mult":
        return torch.bmm(source_encoded, target_encoded.transpose(1, 2))
    if similarity_init == "l2":
        diff = torch.cdist(source_encoded,target_encoded)
        return diff.max() - diff
    if similarity_init == "negative_l2":
        diff = -torch.cdist(source_encoded,target_encoded)
        return diff
    if similarity_init == "difference_exp":
        dist = torch.cdist(source_encoded.contiguous(), target_encoded.contiguous())
        return torch.exp(-dist * 2 * source_encoded.shape[-1])
    if similarity_init == "difference_inverse":
        # TODO maybe (max - tensor) instead of 1/tensor ?
        EPS = 1e-6
        return 1 / (torch.cdist(source_encoded.contiguous(), target_encoded.contiguous()) + EPS)
    if similarity_init == "difference_max_norm":
        dist = torch.cdist(source_encoded.contiguous(), target_encoded.contiguous())
        return (dist.max() - dist) / dist.max()
    if similarity_init == "multiplication":
        return torch.bmm(source_encoded, target_encoded.transpose(1, 2))

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k, idx=None, only_intrinsic=False, permute_feature=True):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    else:
        if(len(idx.shape)==2):
            idx = idx.unsqueeze(0).repeat(batch_size,1,1)
        idx = idx[:, :, :k]
        k = min(k,idx.shape[-1])

    num_idx = idx.shape[1]

    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.contiguous()
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(
        2, 1
    ).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_idx, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    if only_intrinsic is True:
        feature = feature - x
    elif only_intrinsic == 'neighs':
        feature = feature
    elif only_intrinsic == 'concat':
        feature = torch.cat((feature, x), dim=3)
    else:
        feature = torch.cat((feature - x, x), dim=3)

    if permute_feature:
        feature = feature.permute(0, 3, 1, 2).contiguous()

    return feature

def get_s_t_topk(P, k, s_only=False,nn_idx=None):
    """
    Get nearest neighbors per point (similarity value and index) for source and target shapes

    Args:
        P (BxNsxNb Tensor): Similarity matrix
        k: number of neighbors per point
    """
    if(nn_idx is not None):
        assert s_only, "Only for self-construction currently"
        s_nn_idx = nn_idx
        s_nn_val = P.gather(dim=2,index=nn_idx)
        t_nn_val = t_nn_idx = None
    else:
        s_nn_val, s_nn_idx = P.topk(k=min(k,P.shape[2]), dim=2)

        if not s_only:
            t_nn_val, t_nn_idx = P.topk(k=k, dim=1)

            t_nn_val = t_nn_val.transpose(2, 1)
            t_nn_idx = t_nn_idx.transpose(2, 1)
        else:
            t_nn_val = None
            t_nn_idx = None

    return s_nn_val, s_nn_idx, t_nn_val, t_nn_idx

def normalize_P(P, p_normalization, dim=None):
    """
    The method to normalize the P matrix to be "like" a statistical matrix.

    Here we assume that P is Ny times Nx, according to coup paper the columns (per x) should be statistical, hence normalize column wise
    """
    if dim is None:
        dim = 1 if len(P.shape) == 3 else 0

    if p_normalization == "no_normalize":
        return P
    if p_normalization == "l1":
        return F.normalize(P, dim=dim, p=1)
    if p_normalization == "l2":
        return F.normalize(P, dim=dim, p=2)
    if p_normalization == "softmax":
        return F.softmax(P/0.1, dim=dim)
    raise NameError

def get_s_t_neighbors(k, P, sim_normalization, s_only=False, ignore_first=False,nn_idx=None):

    s_nn_sim, s_nn_idx, t_nn_sim, t_nn_idx = get_s_t_topk(P, k, s_only=s_only,nn_idx=nn_idx)
    if ignore_first:
        s_nn_sim, s_nn_idx = s_nn_sim[:, :, 1:], s_nn_idx[:, :, 1:]

    s_nn_weight = normalize_P(s_nn_sim, sim_normalization, dim=2)


    if not s_only:
        if ignore_first:
            t_nn_sim, t_nn_idx = t_nn_sim[:, :, 1:], t_nn_idx[:, :, 1:]

        t_nn_weight = normalize_P(t_nn_sim, sim_normalization, dim=2)
    else:
        t_nn_weight = None

    return s_nn_weight, s_nn_sim, s_nn_idx, t_nn_weight, t_nn_sim, t_nn_idx

class HMRModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = PointTransformerV3(in_channels=3, enable_flash=False)
        self.PoseModel = SMPL(gender='neutral')

        self.selcted_index= np.load("data/preprocess_data/farthest_point_sample_index_0.npy", allow_pickle=True).item()[1024]

        self.template = uni_utils.get_stand_pose_smpl()[0][:,self.selcted_index]
        self.template=self.template- torch.mean(self.template,dim=1).unsqueeze(1)
        self.global_down_sample=nn.MaxPool1d(4)

        self.down_sample=TransitionDown(k=256,nneighbor=5,channels=[259,256,256])

        self.coord_embedding = nn.Sequential(
            nn.Linear(3, 256),
        )

        self.k=10

        conv1 = nn.Sequential()
        conv1.add_module('conv0', torch.nn.Conv1d(256, 128, kernel_size=1))
        conv1.add_module('batchnorm0', nn.BatchNorm1d(128))
        conv1.add_module('relu0', nn.ReLU(inplace=True))
        #
        conv1.add_module('conv1', torch.nn.Conv1d(128, 128, kernel_size=1))
        conv1.add_module('batchnorm1', nn.BatchNorm1d(128))
        conv1.add_module('relu1', nn.ReLU(inplace=True))
        conv1.add_module('conv2', torch.nn.Conv1d(128, 64, kernel_size=1))
        conv1.add_module('batchnorm2', nn.BatchNorm1d(64))
        conv1.add_module('relu2', nn.ReLU(inplace=True))
        conv1.add_module('conv3', torch.nn.Conv1d(64, 64, kernel_size=1))
        self.conv1 = conv1

        self.compress_fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 26),
        )

        self.pose_fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 144),
        )

        self.shape_fc = nn.Sequential(
            nn.Linear(64, 10),
        )
        self.trans_fc = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )
        self.project_fc = nn.Sequential(
            nn.Linear(256,256),
            nn.ReLU(),
            nn.Linear(256,256),)
       # self.time_list=[]

    def reconstruction(self,pos, nn_idx, nn_weight, k):

        # input is BxNx3 根据nn_idx 挑选出 BxNxkx3
        nn_pos = get_graph_feature(pos.transpose(1, 2), k=k, idx=nn_idx, only_intrinsic='neighs', permute_feature=False)

        nn_weighted = nn_pos * nn_weight.unsqueeze(dim=3)
        recon = torch.sum(nn_weighted, dim=2)
        recon_hard = nn_pos[:, :, 0, :]
        return recon, recon_hard

    def forward_source_target(self, source, target):

        # measure cross similarity
        P_non_normalized = measure_similarity("cosine",source["dense_output_features"],
                                                               target["dense_output_features"])

        idx = torch.randperm(target["dense_output_features"].shape[0])
        new_output_f=target["dense_output_features"].clone()[idx]
        source["randperm_weight"] = measure_similarity("cosine",source["dense_output_features"],new_output_f)

        temperature = None

        norm_temperature=0.1

        P_normalized=torch.nn.functional.log_softmax(P_non_normalized/norm_temperature,dim=1)+torch.nn.functional.log_softmax(P_non_normalized/norm_temperature, dim=2)

        threshold=0
        if(torch.mean(torch.max(P_non_normalized, dim=2)[0])<threshold):
             threshold=-1
        source["all_weight"]=P_non_normalized
        source["no_noisy_mask"] = torch.max(P_non_normalized, dim=2)[0] >threshold
        # cross nearest neighbors and weights
        source["cross_nn_weight"], source["cross_nn_sim"], source["cross_nn_idx"], target["cross_nn_weight"], target[
            "cross_nn_sim"], target["cross_nn_idx"] = \
            get_s_t_neighbors(self.k, P_normalized,
                              sim_normalization="softmax")

        source["cross_recon"], source["cross_recon_hard"] = self.reconstruction(source["pos"], target["cross_nn_idx"],
                                                                                target["cross_nn_weight"],
                                                                                self.k)

        target["cross_recon"], target["cross_recon_hard"] = self.reconstruction(target["pos"], source["cross_nn_idx"],
                                                                                source["cross_nn_weight"],
                                                                                self.k)

        return source, target, P_normalized, temperature

    def forward_shape(self, shape):

        P_self = measure_similarity("cosine", shape["dense_output_features"], shape["dense_output_features"])
        nn_idx = None
        shape["self_nn_weight"], _, shape["self_nn_idx"], _, _, _ = \
            get_s_t_neighbors(self.k + 1, P_self, sim_normalization="softmax", s_only=True, ignore_first=True,nn_idx=nn_idx)

        # self reconstruction
        shape["self_recon"], _ = self.reconstruction(shape["pos"], shape["self_nn_idx"], shape["self_nn_weight"], self.k)

        return shape, P_self

    def pose_forward(self,source_data,key="source",dict_key="recon"):
        input_x = source_data[key]["pos"]

        new_xyz,new_features,_=self.down_sample(input_x,source_data[key]["final_feature"])

        new_xyz_embedding=self.coord_embedding(new_xyz)
        data = new_xyz_embedding+new_features
        data = data.permute(0, 2, 1)
        points = self.conv1(data)

        down_sample_global_feature=self.global_down_sample(source_data[key]["global_feature"]).unsqueeze(2)

        points =points+down_sample_global_feature

        pred_kp = self.compress_fc(points)

        pred_kp = pred_kp.permute(0, 2, 1)

        pose_param=self.pose_fc(pred_kp[:,:-2].reshape(input_x.shape[0],-1)).reshape(input_x.shape[0],-1,6)

        pose_param=trans3d.matrix_to_axis_angle(trans3d.rotation_6d_to_matrix(pose_param))

        trans_param=self.trans_fc(pred_kp[:,-2])
        shape_param = self.shape_fc(pred_kp[:,-1])
        shape_param[:, 1:] = 0

        poses_param = pose_param.reshape(points.shape[0], -1).contiguous()

        pred_mesh, pred_kp = self.PoseModel(poses_param, shape_param, trans_param)

        source_data[dict_key]["pred_mesh"] = pred_mesh
        source_data[dict_key]["pred_kp"] = pred_kp
        source_data[dict_key]["pred_param"] = torch.cat((poses_param, shape_param, trans_param), dim=1)


    def new_compute_deep_features(self, data,is_direct=False):

        tmp_template=self.template.clone()
        template_size=tmp_template.shape[0]

        input_x = data["source"]["pos"]
        input_x=torch.cat((tmp_template,input_x),dim=0)
        input_x=torch.cat((torch.zeros(input_x.shape[0],1,3).cuda(),input_x),dim=1)

        output_features = self.encoder.forward_per_point(
            input_x, start_neighs=None
        )


        data["source"]["global_feature"] = output_features[template_size:, 0]
        data["source"]["final_feature"]=output_features[template_size:,1:]

        new_feautre=self.project_fc(output_features)+output_features

        template_output_features=new_feautre[0,1:]
        data["target"]["dense_output_features"]=template_output_features.unsqueeze(0).repeat(data["source"]["pos"].shape[0],1,1)
        data["target"]["dense_output_features"]+=torch.rand(data["target"]["dense_output_features"].shape).cuda()

        data["source"]["dense_output_features"]=new_feautre[template_size:,1:]

        return data

    def forward(self,data):

        data = self.new_compute_deep_features(data, is_direct=False)

        # predict the SMPL-based mesh
        self.pose_forward(data)

        self.distance_weight = False

        data["source"], data["target"], data["P_normalized"], data["temperature"] = self.forward_source_target(
            data["source"], data["target"])

        _, P_self_source = self.forward_shape(data["source"])
        _, P_self_target = self.forward_shape(data["target"])
        data["source"]["self_all_weight"] = P_self_source
        data["target"]["self_all_weight"] = P_self_target

        return data