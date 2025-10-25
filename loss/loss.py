import time
from typing import DefaultDict

import matplotlib.pyplot as plt

import torch
import numpy as np

from utils import universal_utils as uni_utils
from scipy.optimize import linear_sum_assignment
import sys
from collections import defaultdict
import os
from utils.prior_utils import MaxMixturePrior
from model.smplh.pytorch.smpl_layer import SMPL_Layer as SMPLH

def compute_similarity_transform_batch_torch(S1, S2):
    if(S1.shape[1] != 3 ):
        S1 = S1.transpose(1, 2)
        S2 = S2.transpose(1, 2)
    mu1 = S1.mean(dim=2, keepdim=True)
    mu2 = S2.mean(dim=2, keepdim=True)
    X1 = S1 - mu1
    X2 = S2 - mu2

    var1 = torch.sum(X1 ** 2, dim=(1, 2))

    K = torch.bmm(X1, X2.transpose(1, 2))

    U, S, Vh = torch.linalg.svd(K)
    V = Vh.mH

    det = torch.det(torch.bmm(U, V))  # (B,)
    Z = torch.eye(3, device=S1.device, dtype=S1.dtype)[None].repeat(S1.shape[0], 1, 1)
    Z[:, -1, -1] = torch.sign(det)
    R = torch.bmm(torch.bmm(V, Z), U.transpose(1, 2))

    scale = torch.einsum('bii->b', torch.bmm(R, K)) / var1

    t = mu2 - scale[:, None, None] * torch.bmm(R, mu1)

    S1_hat = scale[:, None, None] * torch.bmm(R, S1) + t

    return S1_hat.transpose(1, 2)  # 恢复为 (B, N, 3) 形状


def pa_mpjpe(predicted, target, reduction='mean'):
    aligned_pred = compute_similarity_transform_batch_torch(predicted, target)
    error = torch.norm(aligned_pred - target, dim=-1).mean(dim=-1)

    if reduction == 'mean':
        return error.mean()
    elif reduction == 'sum':
        return error.sum()
    elif reduction == 'none':
        return error
    else:
        raise ValueError(f"Invalid reduction: {reduction}")

def self_recon_loss(source,target,result_data,is_template=False):
    dist=torch.nn.functional.pairwise_distance(source,target,p=2)
    mask=result_data["source"]["valid_mask"]

    if(is_template or torch.sum(~mask)==0):
        return torch.mean(torch.mean(dist,dim=1))
    return torch.mean(dist[mask])


def source_dense_recon_loss(result_data):

    source=result_data["source"]["pos"]
    target=result_data["source"]["cross_recon"]
    cross_idx =result_data["source"]["cross_nn_idx"]
    mask=result_data["source"]["valid_mask"]

    t_cross_idx=cross_idx[:,:,0]

    dense_corr_target=uni_utils.index_points(target,t_cross_idx)

    dist=torch.nn.functional.pairwise_distance(source,dense_corr_target,p=2)

    no_zero_dist=dist[mask]

    target_cross_weight=result_data["target"]["cross_nn_weight"]
    num=512
    _, selected_template_index = target_cross_weight[:, :, 0].topk(num, dim=1)
    selected_source_cross_recon=uni_utils.index_points(target,selected_template_index)
    chamfer_loss=0

    for i in range(0,mask.shape[0]):
        chamfer_loss+=uni_utils.my_chamfer_distance(source[i][mask[i]],selected_source_cross_recon[i][:num])

    chamfer_loss=chamfer_loss/mask.shape[0]+torch.mean(no_zero_dist)

    return chamfer_loss

def template_dense_recon_loss(result_data):

    template=result_data["target"]["pos"]
    template_recon=result_data["target"]["cross_recon"]
    target_cross_weight=result_data["target"]["cross_nn_weight"]
    mask=result_data["source"]["valid_mask"]

    num=512
    _, selected_template_index = target_cross_weight[:, :, 0].topk(512, dim=1)
    selected_template=uni_utils.index_points(template,selected_template_index)

    chamfer_loss=0
    for i in range(0,mask.shape[0]):
        chamfer_loss+=uni_utils.my_chamfer_distance(template_recon[i][mask[i]],selected_template[i][:num])

    return chamfer_loss/mask.shape[0]

def distance_invarance_loss(result_data):

    source=result_data["source"]["pos"]
    target=result_data["target"]["pos"]
    source_self_idx=result_data["source"]["self_nn_idx"]
    source_cross_idx= result_data["source"]["cross_nn_idx"]

    batch_size=source.shape[0]
    the_dis_invar_num=5
    mask=result_data["source"]["valid_mask"]
    mask=mask.unsqueeze(2).repeat(1,1,the_dis_invar_num)

    source_end_index=source_self_idx[:,:,:the_dis_invar_num].reshape(batch_size,-1)

    source_end=uni_utils.index_points(source,source_end_index)
    source_end=source_end.reshape(batch_size,-1,the_dis_invar_num,3)
    t_source=source.unsqueeze(2).repeat(1,1,the_dis_invar_num,1)
    dist=torch.nn.functional.pairwise_distance(t_source[mask],source_end[mask],p=2)


    target_begin=uni_utils.index_points(target,source_cross_idx[:,:,0]).unsqueeze(2).repeat(1,1,the_dis_invar_num,1)
    target_end=uni_utils.index_points(target,uni_utils.index_points(source_cross_idx,source_end_index)[:,:,0]).reshape(batch_size,-1,the_dis_invar_num,3)
    target_dist=torch.nn.functional.pairwise_distance(target_begin[mask],target_end[mask],p=2)

    return torch.mean(torch.abs(dist-target_dist))

def source_recon_mesh_loss(result_data):
    source=result_data["source"]["pos"]
    target=result_data["recon"]["pred_mesh"][:,result_data["selected_index"]]

    cross_idx =result_data["source"]["cross_nn_idx"]
    mask=result_data["source"]["valid_mask"]

    t_cross_idx=cross_idx[:,:,0]

    dense_corr_target=uni_utils.index_points(target,t_cross_idx)

    t_cross_idx2=cross_idx[:,:,0]
    dense_corr_target2=uni_utils.index_points(target,t_cross_idx2)

    partial_recon_mesh_chamfer=0
    second_recon_mesh_chamfer=0
    for i in range(mask.shape[0]):
        partial_recon_mesh_chamfer+=uni_utils.my_chamfer_distance(dense_corr_target[i],source[i][mask[i]])
        second_recon_mesh_chamfer+=uni_utils.my_chamfer_distance(dense_corr_target2[i],source[i][mask[i]])
    partial_recon_mesh_chamfer=(partial_recon_mesh_chamfer*10+second_recon_mesh_chamfer*5)/mask.shape[0]

    return partial_recon_mesh_chamfer


def autoencoder_loss_function(cfg, result_data):

    source_pos = result_data["source"]["pos"]

    batch_size = source_pos.shape[0]

    T = uni_utils.index_points(result_data["recon"]["pred_mesh"],result_data["selected_index"].unsqueeze(0).repeat(batch_size, 1))

    source_pos_mask=result_data["source"]["valid_mask"]
    corr_calc_num =5
    corr_point = uni_utils.index_points(T, result_data["source"]["cross_nn_idx"][:, :, :corr_calc_num].reshape(
        batch_size, -1)).reshape(batch_size, -1, corr_calc_num, 3)

    all_source_pos = source_pos[source_pos_mask]
    to_be_calc_corr_point = corr_point[source_pos_mask]
    the_distance = torch.nn.functional.pairwise_distance(all_source_pos.unsqueeze(1).expand(-1, corr_calc_num, -1),
                                                         to_be_calc_corr_point, p=2)
    the_weight = result_data["source"]["cross_nn_weight"][:, :, :corr_calc_num][source_pos_mask]
    normalized_weight = 3 * (the_weight - torch.min(the_weight, dim=1)[0].unsqueeze(1)) / (
                1e-6 + torch.max(the_weight, dim=1)[0] - torch.min(the_weight, dim=1)[0]).unsqueeze(1)
    the_weight = torch.nn.functional.softmax(normalized_weight, dim=1) + 0.1
    corr_recon_loss = torch.mean(torch.sum(torch.mul(the_distance, the_weight), dim=1))
    return corr_recon_loss

def get_corr_loss(result_data):
    tmp_mask=(torch.sum(torch.abs(result_data["source"]["pos"]),dim=2)>0).flatten()
    score_matrix=result_data["source"]["all_weight"].reshape(-1, 1024)
    corr_loss=torch.nn.functional.cross_entropy(score_matrix*100, result_data["recon"]["pred_segment"].flatten(), reduction='none')
    corr_loss=corr_loss[tmp_mask]
    threshold = torch.quantile(corr_loss, 0.9)
    weight=1 / (torch.exp((corr_loss[corr_loss>threshold] - threshold))).detach()
    weight[weight<2e-2]=2e-2
    corr_loss[corr_loss>threshold]*=weight
    corr_loss=torch.mean(corr_loss)
    return corr_loss

def calc_kp_metrics(cfg,result_data):
    if(cfg["dataset"]["dataname"]=="surreal" or cfg["dataset"]["dataname"]=="humman" or cfg["dataset"]["dataname"]=="sloper4d" or cfg["dataset"]["dataname"]=="behave"):
        pred_kp = result_data["recon"]["pred_kp"]
        flip_pred_kp =pred_kp[:, cfg["dataloader"]["flip_star_kp_index"]]
    else:
        pred_kp = result_data["recon"]["pred_kp"][:,cfg["dataloader"]["aligned_star_kp_index"]]

        flip_pred_kp = result_data["recon"]["pred_kp"][:, cfg["dataloader"]["flip_aligned_star_kp_index"]]

    kp_gap = torch.nn.functional.pairwise_distance(pred_kp,
                                                   result_data["recon"]["label_kp"], p=2)

    pkp=pa_mpjpe(pred_kp,result_data["recon"]["label_kp"], reduction='mean')

    flip_kp_gap = torch.nn.functional.pairwise_distance(flip_pred_kp, result_data["recon"]["label_kp"], p=2)

    gt_mask = torch.sum(torch.abs(result_data["recon"]["label_kp"]), dim=2) > 0

    kp_loss = torch.mean(kp_gap[gt_mask])

    kp_gap[~gt_mask] = 0
    flip_kp_gap[~gt_mask] = 0

    kp_mean = torch.mean(kp_gap, 1)

    flip_kp_mean = torch.mean(flip_kp_gap, 1)
    kp_gap[kp_mean > flip_kp_mean] = flip_kp_gap[kp_mean > flip_kp_mean]

    fkp_loss = torch.mean(kp_gap[gt_mask])

    result_dict = {
        "kp": kp_loss,
        "fkp": fkp_loss,
        "pkp": pkp,
    }

    return result_dict



def gtmesh_correct_gender(source_pos,pose_param,shape_param,transed_joints3d):
    transed_joints3d = uni_utils.surreal_coordinate_transform(transed_joints3d)
    d, j = uni_utils.get_smpl_result_no_humanmodel(pose_param, shape_param, torch.zeros(pose_param.shape[0], 3).cuda(),
                                                   "male", model_root='model/smpl/pytorch/models/')
    rot_pose = j[:, 0, :]
    rot_joints = transed_joints3d[:, 0].cuda()

    trans = rot_joints - rot_pose
    trans = trans.unsqueeze(1)
    d = d + trans
    j = j + trans

    d2, j2 = uni_utils.get_smpl_result_no_humanmodel(pose_param, shape_param,
                                                     torch.zeros(pose_param.shape[0], 3).cuda(), "female",
                                                     model_root='model/smpl/pytorch/models/')
    rot_pose2 = j2[:, 0, :]
    rot_joints2 = transed_joints3d[:, 0].cuda()

    trans2 = rot_joints2 - rot_pose2
    trans2 = trans2.unsqueeze(1)

    d2 = d2 + trans2
    j2 = j2 + trans2

    d = uni_utils.surreal_coordinate_transform(d)

    d2 = uni_utils.surreal_coordinate_transform(d2)

    tmp_pos = source_pos
    c_d1, _ = torch.min(torch.cdist(tmp_pos, d), dim=2)

    c_d2, _ = torch.min(torch.cdist(tmp_pos, d2), dim=2)

    new_cd1 = torch.mean(c_d1, dim=1)
    new_cd2 = torch.mean(c_d2, dim=1)

    mask = new_cd1 > new_cd2
    d[mask] = d2[mask]
    j[mask] = j2[mask]
    j=uni_utils.surreal_coordinate_transform(j)

    return d,j

def corr_idx_loss_function(t_mask,target_pos,predict_corr_idx,gt_corr_id):
    gt_corr_idx=gt_corr_id.clone()
    all_predict_corr_pos=uni_utils.index_points(target_pos,predict_corr_idx)

    all_gt_corr_pos = uni_utils.index_points(target_pos, gt_corr_idx)

    all_gt_corr_pos = all_gt_corr_pos[t_mask]
    all_gt_corr_pos = all_gt_corr_pos.reshape(-1, 3)

    all_predict_corr_pos=all_predict_corr_pos[t_mask].reshape(-1,3)

    predict_no_zero_mask=torch.sum(torch.abs(all_predict_corr_pos),dim=1)>0

    dist=torch.nn.functional.pairwise_distance(all_predict_corr_pos,all_gt_corr_pos,p=2)

    return torch.mean(dist[predict_no_zero_mask])

def surreal_get_gt_mesh_joint(result_data):

    pose_param = result_data["recon"]["pose_param"]
    shape_param = result_data["recon"]["shape_param"]
    transed_joints3d = result_data["recon"]["label_kp"].clone()
    source_pos = result_data["source"]["pos"]

    pose_param = pose_param.cuda()
    shape_param = shape_param.cuda()

    d,j=gtmesh_correct_gender(source_pos,pose_param,shape_param,transed_joints3d)

    gt_mesh = d
    gt_joints=j

    return gt_mesh,gt_joints

def get_corr_error(input_dict,gt_mesh,recon_mesh):
    source_pos=input_dict["source"]["pos"]
    t_mask=input_dict["source"]["valid_mask"]
    try:
        pred_idx= input_dict["source"]["cross_nn_idx"][:,:,0]
    except Exception as e:
        pred_idx = uni_utils.get_corr_idx(source_pos, torch.tensor(recon_mesh).cuda())
    corr_dist=torch.cdist(source_pos,gt_mesh,p=2)

    corr_idx=torch.argmin(corr_dist,dim=2)

    corr_loss = corr_idx_loss_function(t_mask, gt_mesh,
                                       pred_idx, corr_idx)
    return corr_loss


def humman_get_gt_mesh(input_dict):
    pose_param = input_dict["recon"]["pose_param"]
    shape_param = input_dict["recon"]["shape_param"]

    trans_param = input_dict["recon"]["trans_param"]-input_dict["source"]["offset"].squeeze()

    d, j = uni_utils.get_smpl_result_no_humanmodel(pose_param, shape_param, trans_param.cuda(), "neutral",
                                                   model_root='model/smpl/pytorch/models/')
    return d

def sloper4d_get_gt_mesh(input_dict):
    pose_param = input_dict["recon"]["pose_param"]
    shape_param = input_dict["recon"]["shape_param"]
    trans_param=input_dict["recon"]["trans_param"]

    d,j=uni_utils.get_smpl_result_no_humanmodel(pose_param,shape_param,trans_param)

    d=uni_utils.coordinate_transform(d)-input_dict["source"]["offset"]

    return d

def behave_get_gt_mesh(input_dict):
    pose_param = input_dict["recon"]["pose_param"]
    shape_param = input_dict["recon"]["shape_param"]

    trans_param = input_dict["recon"]["trans_param"]-input_dict["source"]["offset"].squeeze()
    if(input_dict["gender"]=="male"):
        smplh = SMPLH(center_idx=0, gender="male", num_betas=10,
                       model_root="model/smplh/native/models/", hands=True).cuda()
    else:
        smplh = SMPLH(center_idx=0, gender="female", num_betas=10,
                             model_root="model/smplh/native/models/", hands=True).cuda()
    d,j,_,_=smplh(pose_param, th_betas=shape_param, th_trans=trans_param)
    return d

class LossModel():
    """Implementation of SMPLify, use surface."""

    def __init__(self,cfg):

        self.cfg=cfg
        self.recoder=defaultdict(list)
        self.max_error=100
        self.save_path = ""
        self.epoch_save_path = ""
        self.pose_prior = MaxMixturePrior(prior_folder="data/preprocess_data/",
                                          num_gaussians=8,
                                          dtype=torch.float32).cuda()


    def calc_corr_loss(self,input_dict,is_test=False):
        result_loss={
            "s_self_re":self_recon_loss(input_dict["source"]["pos"],input_dict["source"]["self_recon"],input_dict,is_template=False),
            "t_self_re":self_recon_loss(input_dict["target"]["pos"],input_dict["target"]["self_recon"],input_dict,is_template=True),
            "s_c_re":source_dense_recon_loss(input_dict),
            "t_c_re":template_dense_recon_loss(input_dict),
            "dis_i":distance_invarance_loss(input_dict),
        }
        if(not is_test):
            result_loss["corr_l"]=get_corr_loss(input_dict)
        return result_loss


    def calc_recon_loss(self,epoch,input_dict):
        result_loss={
            "pr_re":source_recon_mesh_loss(input_dict),
        }
        result_loss["corr_re"]=autoencoder_loss_function(self.cfg,input_dict)
        if(epoch<1):
            result_loss["kp_r"]=torch.mean(torch.abs(input_dict["recon"]["pred_param"][:,:72]))*5

        return result_loss

    def calc_mesh_metrics(self,input_dict,is_eval_mesh=False):
        if(self.cfg["dataset"]["dataname"]=="surreal" ):
            gt_mesh,gt_joint=surreal_get_gt_mesh_joint(input_dict)

        elif(self.cfg["dataset"]["dataname"]=="humman"):
            gt_mesh=humman_get_gt_mesh(input_dict)
        elif(self.cfg["dataset"]["dataname"]=="sloper4d"):
            gt_mesh =sloper4d_get_gt_mesh(input_dict)

        elif(self.cfg["dataset"]["dataname"]=="behave"):
            gt_mesh =behave_get_gt_mesh(input_dict)

        input_dict["recon"]["gt_mesh"]=gt_mesh
        pred_mesh=input_dict["recon"]["pred_mesh"]

        result_loss={
             "c_l": get_corr_error(input_dict, gt_mesh[:,input_dict["selected_index"]], pred_mesh[:,input_dict["selected_index"]]),
        }
        if(is_eval_mesh):
            abs_diff = torch.nn.functional.pairwise_distance(pred_mesh ,gt_mesh)
            loss = torch.mean(abs_diff)
            result_loss["r_m"] = loss

        return result_loss

    def calc_metrics(self,input_dict,is_test):
        kp_loss = calc_kp_metrics(self.cfg, input_dict)
        if(self.cfg["dataset"]["dataname"]=="surreal" or self.cfg["dataset"]["dataname"]=="humman" or self.cfg["dataset"]["dataname"]=="sloper4d" or self.cfg["dataset"]["dataname"]=="behave"):
            mesh_loss=self.calc_mesh_metrics(input_dict,is_test)
            result_loss=dict(mesh_loss,**kp_loss)
        else:
            result_loss=kp_loss

        return result_loss

    def aggregate_loss(self,loss_dict):
        final_loss=0
        for item in loss_dict.keys():
            final_loss+=loss_dict[item]
        return final_loss



    def visualize_loss(self, epoch, batch_idx, total_batch, loss, all_loss_dict,is_test=False):

        if(is_test):
            output_str = '\rT:%d:[%d / %d], ' % (epoch, batch_idx + 1, total_batch)
            output_str += f'loss: %.4f' % loss
        else:
            output_str = '\r%d:[%d / %d], ' % (epoch, batch_idx + 1, total_batch)
            output_str += f'loss: %.4f' % loss


        for loss_name in all_loss_dict.keys():
            output_str += ', '
            try:
                output_str += f'{loss_name}: %.3f' % all_loss_dict[loss_name]
                self.recoder[loss_name].append(all_loss_dict[loss_name].cpu().detach().numpy())
            except Exception as e:
                continue
        sys.stdout.write(output_str)
        sys.stdout.flush()

    def save_model(self,mean_kp,epoch,model):
        if (mean_kp < self.max_error):
            self.max_error = mean_kp
            try:
                os.remove(self.save_path)
            except Exception:
                pass

            self.save_path = "ckpt/" + str(epoch) + "_" + str(round(mean_kp, 3)) + ".pkl"
            torch.save(model, self.save_path)
            print("Save model")
        if (epoch % 2 == 0):
            try:
                os.remove(self.epoch_save_path)
            except Exception:
                pass
            self.epoch_save_path = "ckpt/" + str(epoch) + "_" + str(round(mean_kp, 3)) + ".pkl"
            torch.save(model, self.epoch_save_path)

    def visual_epoch_info_and_save(self,epoch,model=None,is_test=False):
        if(is_test):
            mean_kp = round(np.mean(self.recoder["kp"]), 3)

        output_str = f'\nSummary '
        for loss_name in self.recoder.keys():
            output_str += ', '
            try:
                output_str += f'{loss_name}: %.3f' % np.mean(self.recoder[loss_name])
                self.recoder[loss_name]=[]
            except Exception as e:
                continue
        output_str+="\n"
        sys.stdout.write(output_str)
        sys.stdout.flush()

        if(is_test):
            self.save_model(mean_kp,epoch,model)



    def calc_loss(self,epoch,batch_idx,total_batch,input_dict,_,is_test=False):

        corr_loss=self.calc_corr_loss(input_dict,is_test)
        recon_loss=self.calc_recon_loss(epoch,input_dict)
        kp_loss=self.calc_metrics(input_dict,is_test)


        bp_loss=self.aggregate_loss(corr_loss)+self.aggregate_loss(recon_loss)

        self.recoder["loss"].append(bp_loss.cpu().detach().numpy())
        visualize_loss_item = dict(
            dict(corr_loss, **recon_loss),
            **kp_loss)
        self.visualize_loss(epoch,batch_idx,total_batch,bp_loss,visualize_loss_item,is_test)
        return bp_loss



