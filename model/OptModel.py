"""
Modified Work Based on HMRN
===========================================================

**Original Project**: Self-supervised 3D Human Mesh Recovery from Noisy Point Clouds
**Source URL**: https://github.com/wangsen1312/unsupervised3dhuman

Written by Chang Su
"""

import numpy as np
import torch

from utils.prior_utils import MaxMixturePrior,angle_prior
from model.smpl.pytorch.smpl import SMPL
import yaml
import utils.universal_utils as uni_utils
from utils.universal_utils import my_chamfer_distance
import utils.pytorch3d_transforms as trans3d
from scipy.spatial.transform import Rotation as R



class tmpOutput():
    def __init__(self,v,j):
        self.vertices=v
        self.joints=j
    def set(self,v,j):
        self.vertices=v
        self.joints=j

def new_body_fitting_loss_em(body_pose, preserve_pose, betas, preserve_betas, camera_translation,
                             modelVerts, meshVerts, probArray,
                             pose_prior,
                             smpl_output, modelfaces,
                             sigma=100, pose_prior_weight=4.78,
                             shape_prior_weight=5.0, angle_prior_weight=15.2,
                             betas_preserve_weight=10.0, pose_preserve_weight=10.0,
                             chamfer_weight=2000.0,
                             correspond_weight=800.0,
                             point2mesh_weight=5000.0,
                             use_collision=False,
                             model_vertices=None, model_faces=None,
                             search_tree=None, pen_distance=None, filter_faces=None,
                             collision_loss_weight=1000, mask_num=None,delta_gap=None, valid_mask=None
                             ):
    """
    Loss function for body fitting
    """

    correspond_loss = correspond_weight * (probArray * delta_gap)

    correspond_loss=torch.split(correspond_loss,mask_num.tolist())
    correspond_loss=torch.stack([torch.sum(i) for i in correspond_loss])

    pose_prior_loss = (pose_prior_weight ** 2) * pose_prior(body_pose, betas)
    # Angle prior for knees and elbows
    angle_prior_loss = (angle_prior_weight ** 2) * angle_prior(body_pose).sum(dim=-1)

    # Regularizer to prevent betas from taking large values
    shape_prior_loss = (shape_prior_weight ** 2) * (betas ** 2).sum(dim=-1)

    betas_preserve_loss = (betas_preserve_weight ** 2) * ((betas - preserve_betas) ** 2).sum(dim=-1)
    pose_preserve_loss = (pose_preserve_weight ** 2) * ((body_pose - preserve_pose) ** 2).sum(dim=-1)

    dist1, dist2 = my_chamfer_distance(meshVerts, modelVerts, first_dim=1, is_return_dist=True)

    dist2=torch.mean(dist2*valid_mask, dim=-1)
    dist1=torch.mean(dist1,dim=-1)


    chamfer_loss = (chamfer_weight ** 2) * dist1+ ((chamfer_weight / 4.0) ** 2) * dist2

    total_loss =correspond_loss +pose_prior_loss + angle_prior_loss + shape_prior_loss + betas_preserve_loss + pose_preserve_loss + chamfer_loss

    if(correspond_weight<2000):
        loss_item=correspond_loss*10+pose_prior_loss/10+chamfer_loss*10
    else:
        loss_item=torch.div(correspond_loss,torch.tensor(mask_num))*10


    return total_loss.sum(),loss_item


def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))


def adjust_learning_rate(optimizer, learning_rate, i_iter, max_iter, power):
    """Sets the learning rate to the initial LR divided by 5 at 60th, 120th and 160th epochs"""
    lr = lr_poly(learning_rate, i_iter, max_iter, power)
    optimizer.param_groups[0]['lr'] = lr
    return lr




class OptModel():
    """Implementation of SMPLify, use surface."""

    def __init__(self,
                 step_size=1e-1,
                 batch_size=1,
                 num_iters=100,
                 selected_index=np.arange(6890),
                 use_collision=False,
                 device=torch.device('cuda:0'),
                 mu=0.05
                 ):
        # Store options
        self.batch_size = batch_size
        self.device = device
        self.step_size = step_size

        self.num_iters = num_iters
        # GMM pose prior
        self.pose_prior = MaxMixturePrior(prior_folder="data/preprocess_data/",
                                          num_gaussians=8,
                                          dtype=torch.float32).to(device)
        # Load SMPL model
        self.selected_index = np.load("data/preprocess_data/farthest_point_sample_index_0.npy", allow_pickle=True).item()[1024]

        self.use_collision = use_collision

        self.smpl_model=SMPL(gender="neutral").cuda()
        # mu prob
        self.mu = mu


    def new_prob_cal(self, modelVerts, meshVerts, sigma=0.05 ** 2, mu=0.02):
        N = 1024
        batch_size, M, _ = modelVerts.shape

        sum_model = torch.sum(modelVerts ** 2, dim=2, keepdim=True)  # (B, M, 1)
        sum_mesh = torch.sum(meshVerts ** 2, dim=2, keepdim=True).permute(0, 2, 1)  # (B, 1, N)
        dot_product = torch.bmm(modelVerts, meshVerts.permute(0, 2, 1))  # (B, M, N)
        deltaVerts = sum_model + sum_mesh - 2 * dot_product  # (B, M, N)
        with torch.no_grad():


            sigmaInit = sigma
            d = 3.0  # three dimension
            mu_c = ((2.0 * torch.asin(torch.tensor(1.)) * sigmaInit) ** (d / 2.0) * mu * M) / ((1 - mu) * N)

            deltaExp = torch.exp(torch.div(-deltaVerts, (2 * sigmaInit)))
            deltaExpN = torch.reshape(torch.sum(deltaExp, dim=1), (batch_size, 1, N)).expand(-1, M, -1)

            probArray = deltaExp / (deltaExpN + mu_c)

            mask = probArray > 1e-6

            mask = torch.logical_and(mask, self.input_pc_mask.unsqueeze(1).expand(-1, M, -1))

        mask_num=torch.sum(mask,dim=(1,2))
        new_sigma=(probArray*deltaVerts).reshape(batch_size,-1).sum(-1)/probArray.reshape(batch_size,-1).sum(-1)

        new_sigma/=5
        new_sigma=new_sigma.unsqueeze(1).unsqueeze(2)

        return probArray[mask], deltaVerts[mask], new_sigma,mask_num
        # return probInput, modelInd, meshInd

    # ---- get the man function hrere
    def __call__(self, init_pose, init_betas, init_cam_t, meshVerts,corr_weight=1000.0):
        """Perform body fitting.
        Input:
            init_pose: SMPL pose
            init_betas: SMPL betas
            init_cam_t: Camera translation
            meshVerts: point3d from mesh
        Returns:
            vertices: Vertices of optimized shape
            joints: 3D joints of optimized shape
            pose: SMPL pose parameters of optimized shape
            betas: SMPL beta parameters of optimized shape
            camera_translation: Camera translation
        """

        ### add the mesh inter-section to avoid
        search_tree = None
        pen_distance = None
        filter_faces = None

        log_vars = torch.nn.Parameter(torch.zeros(3))
        # self.log_vars = torch.nn.Parameter(torch.ones(num_tasks))
        # Make camera translation a learnable parameter
        # Split SMPL pose to body pose and global orientation
        body_pose = init_pose[:, 3:].detach().clone()

        global_orient = init_pose[:, :3].detach().clone()

        camera_translation = init_cam_t.detach().clone()

        betas = init_betas.detach().clone()
        preserve_betas = init_betas.detach().clone()
        preserve_pose = init_pose[:, 3:].detach().clone()


        betas.requires_grad = True
        body_pose.requires_grad = True
        global_orient.requires_grad = True
        camera_translation.requires_grad = True
        body_opt_params = [body_pose, global_orient, betas, camera_translation,log_vars]  #

        lambda1=lambda epoch: 0.99 ** epoch
        body_optimizer = torch.optim.Adam(body_opt_params, lr=0.05)#
        scheduler=torch.optim.lr_scheduler.LambdaLR(body_optimizer,lr_lambda=lambda1)

        # try:
        self.num_iters =101
        self.input_pc_mask=torch.sum(torch.abs(meshVerts), dim=-1) > 1e-6


        # st = time.perf_counter()
        tmp_sigma = (0.1 ** 2) * (self.num_iters  + 1) / (4 * self.num_iters)
        mini_sigma=(0.1 ** 2) * 1e-2

        for i in range(self.num_iters):
            body_optimizer.zero_grad()

            v, j = self.smpl_model(torch.cat((global_orient, body_pose), dim=1), betas, camera_translation)
            smpl_output = tmpOutput(v, j)
            modelVerts = smpl_output.vertices[:, self.selected_index]

            proArray, delta_gap, tmp_sigma,mask_num= self.new_prob_cal(modelVerts, meshVerts, sigma=tmp_sigma, mu=self.mu)
            tmp_sigma[tmp_sigma<mini_sigma]=mini_sigma



            loss,loss_item = new_body_fitting_loss_em(body_pose, preserve_pose, betas, preserve_betas,
                                             camera_translation,
                                             modelVerts, meshVerts, proArray,
                                             self.pose_prior,
                                             smpl_output, 1,
                                             pose_prior_weight=4.78 * 3.0,
                                             pose_preserve_weight=3.0,
                                             correspond_weight=corr_weight,
                                             chamfer_weight=100.0,
                                             point2mesh_weight=200.0,
                                             shape_prior_weight=2.0,
                                             use_collision=self.use_collision,
                                             model_vertices=smpl_output.vertices, model_faces=None,
                                             search_tree=i, pen_distance=pen_distance,
                                             delta_gap=delta_gap,mask_num=mask_num,valid_mask=self.input_pc_mask)

            loss.backward()

            body_optimizer.step()

        with torch.no_grad():

            v, j = self.smpl_model(torch.cat((global_orient, body_pose), dim=1), betas, camera_translation)
            smpl_output = tmpOutput(v, j)

        vertices = smpl_output.vertices.detach()
        joints = smpl_output.joints.detach()[:, :24]
        pose = torch.cat([global_orient, body_pose], dim=-1).detach()
        betas = betas.detach()
        pose_prior=self.pose_prior(pose[:,3:], betas)

        return vertices, joints, pose, betas, camera_translation,pose_prior,loss_item

    def direction_estimation(self,input_dict):
        B, N, C = input_dict["source"]["pos"].shape

        expand_num = 4
        pose_param, shape_param, trans_param = uni_utils.get_possible_smpl(B, expand_num, use_fix=True)
        aug_data = input_dict["source"]["pos"].clone().unsqueeze(1).expand(-1, expand_num, -1, -1).reshape(-1, N, C)

        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, new_opt_cam_t, pose_prior,loss_item = self(
            pose_param.detach(),
            shape_param.detach(),
            trans_param.detach(),
            aug_data,
        )

        new_opt_vertices = new_opt_vertices.to(torch.float32)

        loss_item=loss_item.reshape(B, expand_num)

        the_corr_idx=torch.argmin(loss_item,dim=1)
        new_opt_vertices = new_opt_vertices.reshape(B, expand_num, 6890, C)
        final_opt_vertices = new_opt_vertices[torch.arange(B), the_corr_idx]

        new_opt_joints = new_opt_joints.reshape(B, expand_num, 24, 3)
        final_opt_joints = new_opt_joints[torch.arange(B), the_corr_idx]
        return final_opt_vertices, final_opt_joints

    def opt_direction_estimation(self,input_dict,pose_param,shape_param,tans_param):
        B, N, C = input_dict["source"]["pos"].shape

        expand_num = 2

        pose_param=pose_param.unsqueeze(1).expand(-1,expand_num,-1).clone()

        if(expand_num>1):

            global_orient = pose_param[:, 0,:3]
            result = trans3d.axis_angle_to_matrix(global_orient)
            rx_180 = R.from_euler('y', 180, degrees=True).as_matrix()

            result = result @ (torch.tensor(rx_180).cuda().to(torch.float32))

            pose_param[:,1]=0
            pose_param[:, 1,:3] = trans3d.matrix_to_axis_angle(result)

        pose_param=pose_param.reshape(B*expand_num,-1)
        shape_param=shape_param.unsqueeze(1).expand(-1,expand_num,-1).reshape(B*expand_num,-1)
        trans_param=tans_param.unsqueeze(1).expand(-1,expand_num,-1).reshape(B*expand_num,-1)

        aug_data = input_dict["source"]["pos"].clone().unsqueeze(1).expand(-1, expand_num, -1, -1).reshape(-1, N, C)

        new_opt_vertices, new_opt_joints, new_opt_pose, new_opt_betas, new_opt_cam_t,pose_prior, loss_item = self(
            pose_param.detach(),
            shape_param.detach(),
            trans_param.detach(),
            aug_data,
            corr_weight=1000*10
        )

        new_opt_vertices = new_opt_vertices.to(torch.float32)

        result = uni_utils.hyperextend_jud(new_opt_joints)


        result = result.reshape(B, expand_num)
        if(expand_num==2):
            the_corr_idx=((result[:,1]-result[:,0])>1.5).to(torch.long)
        else:
            the_corr_idx = torch.argmax(result, dim=1)

        loss_item=loss_item.reshape(B, expand_num)
        the_corr_idx = torch.logical_or(the_corr_idx,loss_item[:,0]>(loss_item[:,1]+3)).to(torch.long)

        new_opt_vertices = new_opt_vertices.reshape(B, expand_num, 6890, C)
        final_opt_vertices = new_opt_vertices[torch.arange(B), the_corr_idx]

        new_opt_joints = new_opt_joints.reshape(B, expand_num, 24, 3)
        final_opt_joints = new_opt_joints[torch.arange(B), the_corr_idx]
        return final_opt_vertices, final_opt_joints

    def init_process(self,data):
        new_opt_vertices = self.direction_estimation(data)
        return new_opt_vertices
