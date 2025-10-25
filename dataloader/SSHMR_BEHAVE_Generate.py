"""
demo code to convert RGBD images to point clouds, using the segmentation mask
example usage:
python tools/rgbd2pclouds.py BEHAVE_SEQ_ROOT/Date05_Sub05_chairwood -t obj

Modified from BEHAVE
===========================================================

**Original Project**: Behave: Dataset and method for tracking human object interactions
**Source URL**: https://github.com/xiexh20/behave-dataset
"""
import sys, os

from torch import cdist

sys.path.append(os.getcwd())
import cv2
import numpy as np
from tqdm import tqdm
from os.path import join, dirname, isfile
from data.frame_data import FrameDataReader
from data.kinect_transform import KinectTransform
from tools.pc_filter import PCloudsFilter
from data.utils import write_pointcloud
import pickle
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from libsmpl.smplpytorch.pytorch.smpl_layer import SMPL_Layer as SMPLH
import json
import torch
import torch.nn.functional as F
import time
import open3d as o3d
import copy

def statistical_fliter(scan):
    vector3d_vector=o3d.utility.Vector3dVector(scan)
    scan = o3d.geometry.PointCloud()
    scan.points = vector3d_vector
    pcd = copy.deepcopy(scan)
    cl, ind = scan.remove_statistical_outlier(nb_neighbors=7, std_ratio=3)
    ground = pcd.select_by_index(ind)

    return np.asarray(ground.points)

def compare_mutil_pc(pc_list, label_list=[], color_list=[None, None, None,None,None], size_list=None, ax=None, is_show=True,is_paper_plot=False,is_cb=False):
    if(len(label_list)==0):
        label_list=[str(i) for i in range(len(pc_list))]
    if (ax == None):
        fig = plt.figure()
        ax = Axes3D(fig)
        fig.add_axes(ax)
    if (size_list == None):
        size_list = [10] * len(label_list)

    delete_zero = True
    for index, data in enumerate(pc_list):
        if (type(data) == torch.Tensor):
            data = data.cpu().detach().numpy()
        #print(index, data.shape)
        if (delete_zero):
            # mask=data[:,2]>1e-6
            mask = np.sum(np.abs(data), axis=1) > 0
            #print(mask.shape)
            # print(data[~mask])

            data = data[mask]
            # if(color_list[index]!=None):
            #     color_list[index]=color_list[index][mask]

        #print(data.shape)
        t=ax.scatter(data[:, 0], data[:, 1], data[:, 2], c=color_list[index], s=size_list[index], label=label_list[index])
        # ax.set_zlim(bottom=np.min(data[:, 2]))
    if(is_paper_plot):
        ax.grid(None)

        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    if(is_cb):
        plt.colorbar(t)
    if (is_show):
        plt.legend()
        plt.show()

def quaternion_to_matrix(quaternions):
    """
    Convert rotations given as quaternions to rotation matrices.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def axis_angle_to_matrix(axis_angle):
    """
    Convert rotations given as axis/angle to rotation matrices.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))

def axis_angle_to_quaternion(axis_angle):
    """
    Convert rotations given as axis/angle to quaternions.

    Args:
        axis_angle: Rotations given as a vector in axis angle form,
            as a tensor of shape (..., 3), where the magnitude is
            the angle turned anticlockwise in radians around the
            vector's direction.

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions

def quaternion_to_axis_angle(quaternions):
    """
    Convert rotations given as quaternions to axis/angle.

    Args:
        quaternions: quaternions with real part first,
            as tensor of shape (..., 4).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # for x small, sin(x/2) is about x/2 - (x/2)^3/6
    # so sin(x/2)/x is about 1/2 - (x*x)/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles

def matrix_to_axis_angle(matrix):
    """
    Convert rotations given as rotation matrices to axis/angle.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        Rotations given as a vector in axis angle form, as a tensor
            of shape (..., 3), where the magnitude is the angle
            turned anticlockwise in radians around the vector's
            direction.
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))
def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    """
    Returns torch.sqrt(torch.max(0, x))
    but with a zero subgradient where x is 0.
    """
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret

def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """
    Convert rotations given as rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices as tensor of shape (..., 3, 3).

    Returns:
        quaternions with real part first, as tensor of shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix  shape f{matrix.shape}.")

    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(*batch_dim, 9), dim=-1
    )

    q_abs = _sqrt_positive_part(
        torch.stack(
            [
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ],
            dim=-1,
        )
    )

    # we produce the desired quaternion multiplied by each of r, i, j, k
    quat_by_rijk = torch.stack(
        [
            torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
            torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
            torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
            torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
        ],
        dim=-2,
    )

    # We floor here at 0.1 but the exact level is not important; if q_abs is small,
    # the candidate won't be picked.
    # pyre-ignore [16]: `torch.Tensor` has no attribute `new_tensor`.
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(q_abs.new_tensor(0.1)))

    # if not for numerical problems, quat_candidates[i] should be same (up to a sign),
    # forall i; we pick the best-conditioned one (with the largest denominator)

    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :  # pyre-ignore[16]
    ].reshape(*batch_dim, 4)


class BEHAVE_Generator():
    def __init__(self):
        self.male_smplh = SMPLH(center_idx=0, gender="male", num_betas=10,
                       model_root="../libsmpl/smplpytorch/native/models/", hands=True)

        self.female_smplh = SMPLH(center_idx=0, gender="female", num_betas=10,
                         model_root="../libsmpl/smplpytorch/native/models/", hands=True)

        self.data_list=[]

    def generate_data(self,out_frame_path,pc_all):
       # print(out_frame_path)
        info_path=dirname(out_frame_path).replace("processed","sequences")+"/info.json"

        info = json.load(open(info_path))
        gender = info["gender"]

        rotation_axis_angles = torch.tensor([[np.pi, 0, 0]])
        local_rotmats = axis_angle_to_matrix(rotation_axis_angles)

        smpl_pkl_path = out_frame_path.replace("processed", "sequences") + "/person/fit02/person_fit.pkl"
        with open(smpl_pkl_path, 'rb') as f:
            smpl_pkl = pickle.load(f, encoding='bytes')

        pose = torch.tensor(smpl_pkl["pose"]).unsqueeze(0)
        betas = torch.tensor(smpl_pkl["betas"]).unsqueeze(0)
        trans = torch.tensor(smpl_pkl["trans"]).unsqueeze(0)

        if (gender == "male"):
            d, j, _, _ = self.male_smplh(pose, th_betas=betas, th_trans=trans)
        else:
            d, j, _, _ = self.female_smplh(pose, th_betas=betas, th_trans=trans)

        trans = torch.bmm(j[0, 0].unsqueeze(0).unsqueeze(1), local_rotmats)[0] - (j[0, 0] - trans)
        pose[:, :3] = matrix_to_axis_angle(
            torch.bmm(local_rotmats, axis_angle_to_matrix(pose[:, :3])))
        if (gender == "male"):
            nd, j, _, _ = self.male_smplh(pose, th_betas=betas, th_trans=trans)
        else:
            nd, j, _, _ = self.female_smplh(pose, th_betas=betas, th_trans=trans)
        j = torch.cat((j[:, :23], j[:, 37].unsqueeze(0)), dim=1)
        threshold_dist=1.3
        for index, item in enumerate(pc_all):
            mean_pos = torch.mean(torch.tensor(item).cuda(), dim=0)

            tmp_index = np.random.randint(0, len(item), 1200)
            tmp_pos=torch.tensor(item[tmp_index]).cuda()
            tmp_dist=torch.cdist(tmp_pos,mean_pos.unsqueeze(0)).flatten()

            dist_matrix=torch.cdist(tmp_pos,tmp_pos)
            top4_dist=torch.topk(dist_matrix,6,dim=1,largest=False).values
            fliter_mask=torch.logical_and(top4_dist[:,5]<0.2,tmp_dist<threshold_dist).cpu()
            #print((tmp_dist<1.5).shape)
            new_index=tmp_index[fliter_mask][:1024]
            #selected_num
            if(len(new_index)<1024):
                print(out_frame_path)
                new_index=np.random.randint(0, len(item), 1800)
                tmp_pos = torch.tensor(item[new_index]).cuda()
                tmp_dist = torch.cdist(tmp_pos, mean_pos.unsqueeze(0)).flatten()
                dist_matrix = torch.cdist(tmp_pos, tmp_pos)
                top4_dist = torch.topk(dist_matrix, 6, dim=1, largest=False).values
                fliter_mask = torch.logical_and(top4_dist[:, 5] < 0.2, tmp_dist < threshold_dist).cpu()

                new_index = new_index[fliter_mask][:1024]

            item = item[new_index]

            pc_data = torch.bmm(torch.tensor(item).unsqueeze(0).to(torch.float32), local_rotmats)

            if(not os.path.exists(out_frame_path)):
                os.makedirs(out_frame_path)

            pc_path=out_frame_path+"/"+str(index)+".pkl"
            with open(pc_path, 'wb') as f:
                pickle.dump(pc_data[0], f)


            tmp_data = {
                "data_dir": pc_path.replace("../../SHMR/","").replace("\\","/"),
                "gt_pose": pose[0].detach().cpu().numpy(),
                "gt_shape": betas[0].detach().cpu().numpy(),
                "camLoc": trans[0].detach().cpu().numpy(),
                "gt_joints3d": j[0].detach().cpu().numpy(),
                "gender": gender
            }
            self.data_list.append(tmp_data)

    def save_data_list(self,path):
        np.save(path, self.data_list)



def main(args,trainGenerator):
    reader = FrameDataReader(args.seq, check_image=False)
    kin_transform = KinectTransform(args.seq)
    rotations, translations = kin_transform.local2world_R, kin_transform.local2world_t
    start = args.start
    end = reader.cvt_end(args.end)
    kids = reader.seq_info.kids

    # specify output dir
    outroot = dirname(args.seq) if args.out is None else args.out
    out_seq = join(outroot, reader.seq_name)
    target = 'person' if args.target == 'person' else reader.seq_info.get_obj_name(convert=True)

    # point cloud processing
    # filter = PCloudsFilter()

    for i in tqdm(range(start, end,2)):

        rgb_imgs = reader.get_color_images(i, kids)
        dmaps = reader.get_depth_images(i, kids)

        out_frame = join(out_seq, reader.frame_time(i))
        out_frame=out_frame.replace("BEHAVE/sequences", "BEHAVE/processed")

        masks = []
        complete = True


        for kid in kids:
            mask = reader.get_mask(i, kid, args.target)
            if mask is None:
                mask = np.zeros((1536, 2048)).astype(bool)
                complete = False
            masks.append(mask)
        if not complete:
            print(f"Warning: the {target} mask for frame {reader.frame_time(i)} is not complete!")
            continue
        pc_all, color_all = [], []

        for kid, mask in enumerate(masks):
            if np.sum(mask) == 0:
                continue
            depth_masked = np.copy(dmaps[kid])  # the rgb mask can be directly applied to depth image
            depth_masked[~mask] = 0

            pc = kin_transform.intrinsics[kid].dmap2pc(depth_masked)
            if len(pc) == 0:
                continue
            pc_world = np.matmul(pc, rotations[kid].T) + translations[kid]


            pc_all.append(pc_world)
        if len(pc_all) == 0:
            print("no mask in", reader.get_frame_folder(i))
            continue
        # do filtering
        np.random.seed(42)

        trainGenerator.generate_data(out_frame,pc_all)

    print('all done')


def get_all_file(file_dir, is_dir=False):
    for root, dirs, files in os.walk(file_dir):
        if (is_dir):
            return dirs
        return files


split_dict={
  "train": [
    "Date01_Sub01_backpack_back",
    "Date01_Sub01_backpack_hand",
    "Date01_Sub01_backpack_hug",
    "Date01_Sub01_basketball",
    "Date01_Sub01_boxlarge_hand",
    "Date01_Sub01_boxlong_hand",
    "Date01_Sub01_boxmedium_hand",
    "Date01_Sub01_boxsmall_hand",
    "Date01_Sub01_boxtiny_hand",
    "Date01_Sub01_chairblack_hand",
    "Date01_Sub01_chairblack_lift",
    "Date01_Sub01_chairblack_sit",
    "Date01_Sub01_chairwood_hand",
    "Date01_Sub01_chairwood_lift",
    "Date01_Sub01_chairwood_sit",
    "Date01_Sub01_keyboard_move",
    "Date01_Sub01_keyboard_typing",
    "Date01_Sub01_monitor_hand",
    "Date01_Sub01_monitor_move",
    "Date01_Sub01_plasticcontainer",
    "Date01_Sub01_stool_move",
    "Date01_Sub01_stool_sit",
    "Date01_Sub01_suitcase",
    "Date01_Sub01_suitcase_lift",
    "Date01_Sub01_tablesmall_lean",
    "Date01_Sub01_tablesmall_lift",
    "Date01_Sub01_tablesmall_move",
    "Date01_Sub01_tablesquare_hand",
    "Date01_Sub01_tablesquare_lift",
    "Date01_Sub01_tablesquare_sit",
    "Date01_Sub01_toolbox",
    "Date01_Sub01_trashbin",
    "Date01_Sub01_yogaball",
    "Date01_Sub01_yogaball_play",
    "Date01_Sub01_yogamat_hand",
    "Date02_Sub02_backpack_back",
    "Date02_Sub02_backpack_hand",
    "Date02_Sub02_backpack_twohand",
    "Date02_Sub02_basketball",
    "Date02_Sub02_boxlarge_hand",
    "Date02_Sub02_boxlong_hand",
    "Date02_Sub02_boxmedium_hand",
    "Date02_Sub02_boxsmall_hand",
    "Date02_Sub02_boxtiny_hand",
    "Date02_Sub02_chairblack_hand",
    "Date02_Sub02_chairblack_lift",
    "Date02_Sub02_chairblack_sit",
    "Date02_Sub02_chairwood_hand",
    "Date02_Sub02_chairwood_sit",
    "Date02_Sub02_keyboard_move",
    "Date02_Sub02_keyboard_typing",
    "Date02_Sub02_monitor_hand",
    "Date02_Sub02_monitor_move",
    "Date02_Sub02_plasticcontainer",
    "Date02_Sub02_stool_move",
    "Date02_Sub02_stool_sit",
    "Date02_Sub02_suitcase_ground",
    "Date02_Sub02_suitcase_lift",
    "Date02_Sub02_tablesmall_lean",
    "Date02_Sub02_tablesmall_lift",
    "Date02_Sub02_tablesmall_move",
    "Date02_Sub02_tablesquare_lift",
    "Date02_Sub02_tablesquare_move",
    "Date02_Sub02_tablesquare_sit",
    "Date02_Sub02_toolbox",
    "Date02_Sub02_trashbin",
    "Date02_Sub02_yogaball_play",
    "Date02_Sub02_yogaball_sit",
    "Date02_Sub02_yogamat",
    "Date04_Sub05_backpack",
    "Date04_Sub05_basketball",
    "Date04_Sub05_boxlarge",
    "Date04_Sub05_boxlong",
    "Date04_Sub05_boxmedium",
    "Date04_Sub05_boxsmall",
    "Date04_Sub05_boxtiny",
    "Date04_Sub05_chairblack",
    "Date04_Sub05_chairwood",
    "Date04_Sub05_keyboard",
    "Date04_Sub05_monitor",
    "Date04_Sub05_monitor_sit",
    "Date04_Sub05_plasticcontainer",
    "Date04_Sub05_stool",
    "Date04_Sub05_suitcase",
    "Date04_Sub05_suitcase_open",
    "Date04_Sub05_tablesmall",
    "Date04_Sub05_tablesquare",
    "Date04_Sub05_toolbox",
    "Date04_Sub05_trashbin",
    "Date04_Sub05_yogaball",
    "Date04_Sub05_yogamat",
    "Date05_Sub05_backpack",
    "Date05_Sub05_chairblack",
    "Date05_Sub05_chairwood",
    "Date05_Sub05_yogaball",
    "Date05_Sub06_backpack_back",
    "Date05_Sub06_backpack_hand",
    "Date05_Sub06_backpack_twohand",
    "Date05_Sub06_basketball",
    "Date05_Sub06_boxlarge",
    "Date05_Sub06_boxlong",
    "Date05_Sub06_boxmedium",
    "Date05_Sub06_boxsmall",
    "Date05_Sub06_boxtiny",
    "Date05_Sub06_chairblack_hand",
    "Date05_Sub06_chairblack_lift",
    "Date05_Sub06_chairblack_sit",
    "Date05_Sub06_chairwood_hand",
    "Date05_Sub06_chairwood_lift",
    "Date05_Sub06_chairwood_sit",
    "Date05_Sub06_keyboard_hand",
    "Date05_Sub06_keyboard_move",
    "Date05_Sub06_monitor_hand",
    "Date05_Sub06_monitor_move",
    "Date05_Sub06_plasticcontainer",
    "Date05_Sub06_stool_lift",
    "Date05_Sub06_stool_sit",
    "Date05_Sub06_suitcase_hand",
    "Date05_Sub06_suitcase_lift",
    "Date05_Sub06_tablesmall_hand",
    "Date05_Sub06_tablesmall_lean",
    "Date05_Sub06_tablesmall_lift",
    "Date05_Sub06_tablesquare_lift",
    "Date05_Sub06_tablesquare_move",
    "Date05_Sub06_tablesquare_sit",
    "Date05_Sub06_toolbox",
    "Date05_Sub06_trashbin",
    "Date05_Sub06_yogaball_play",
    "Date05_Sub06_yogaball_sit",
    "Date05_Sub06_yogamat",
    "Date06_Sub07_backpack_back",
    "Date06_Sub07_backpack_hand",
    "Date06_Sub07_backpack_twohand",
    "Date06_Sub07_basketball",
    "Date06_Sub07_boxlarge",
    "Date06_Sub07_boxlong",
    "Date06_Sub07_boxmedium",
    "Date06_Sub07_boxsmall",
    "Date06_Sub07_boxtiny",
    "Date06_Sub07_chairblack_hand",
    "Date06_Sub07_chairblack_lift",
    "Date06_Sub07_chairblack_sit",
    "Date06_Sub07_chairwood_hand",
    "Date06_Sub07_chairwood_lift",
    "Date06_Sub07_chairwood_sit",
    "Date06_Sub07_keyboard_move",
    "Date06_Sub07_keyboard_typing",
    "Date06_Sub07_monitor_move",
    "Date06_Sub07_plasticcontainer",
    "Date06_Sub07_stool_lift",
    "Date06_Sub07_stool_sit",
    "Date06_Sub07_suitcase_lift",
    "Date06_Sub07_suitcase_move",
    "Date06_Sub07_tablesmall_lean",
    "Date06_Sub07_tablesmall_lift",
    "Date06_Sub07_tablesmall_move",
    "Date06_Sub07_tablesquare_lift",
    "Date06_Sub07_tablesquare_move",
    "Date06_Sub07_tablesquare_sit",
    "Date06_Sub07_toolbox",
    "Date06_Sub07_trashbin",
    "Date06_Sub07_yogaball_play",
    "Date06_Sub07_yogaball_sit",
    "Date06_Sub07_yogamat",
    "Date07_Sub04_backpack_back",
    "Date07_Sub04_backpack_hand",
    "Date07_Sub04_backpack_twohand",
    "Date07_Sub04_basketball",
    "Date07_Sub04_boxlarge",
    "Date07_Sub04_boxlong",
    "Date07_Sub04_boxmedium",
    "Date07_Sub04_boxsmall",
    "Date07_Sub04_boxtiny",
    "Date07_Sub04_chairblack_hand",
    "Date07_Sub04_chairblack_lift",
    "Date07_Sub04_chairblack_sit",
    "Date07_Sub04_chairwood_hand",
    "Date07_Sub04_chairwood_lift",
    "Date07_Sub04_chairwood_sit",
    "Date07_Sub04_keyboard_move",
    "Date07_Sub04_keyboard_typing",
    "Date07_Sub04_monitor_hand",
    "Date07_Sub04_monitor_move",
    "Date07_Sub04_plasticcontainer",
    "Date07_Sub04_stool_lift",
    "Date07_Sub04_stool_sit",
    "Date07_Sub04_suitcase_lift",
    "Date07_Sub04_suitcase_open",
    "Date07_Sub04_tablesmall_lean",
    "Date07_Sub04_tablesmall_lift",
    "Date07_Sub04_tablesmall_move",
    "Date07_Sub04_tablesquare_lift",
    "Date07_Sub04_tablesquare_move",
    "Date07_Sub04_tablesquare_sit",
    "Date07_Sub04_toolbox_lift",
    "Date07_Sub04_trashbin",
    "Date07_Sub04_yogaball_play",
    "Date07_Sub04_yogaball_sit",
    "Date07_Sub04_yogamat",
    "Date07_Sub05_suitcase_lift",
    "Date07_Sub05_suitcase_open",
    "Date07_Sub05_tablesmall",
    "Date07_Sub05_tablesquare",
    "Date07_Sub08_backpack_back",
    "Date07_Sub08_backpack_hand",
    "Date07_Sub08_backpack_hug",
    "Date07_Sub08_basketball",
    "Date07_Sub08_boxlarge",
    "Date07_Sub08_boxlong",
    "Date07_Sub08_boxmedium",
    "Date07_Sub08_boxsmall",
    "Date07_Sub08_boxtiny",
    "Date07_Sub08_chairblack_hand",
    "Date07_Sub08_chairblack_lift",
    "Date07_Sub08_chairblack_sit",
    "Date07_Sub08_chairwood_hand",
    "Date07_Sub08_chairwood_lift",
    "Date07_Sub08_chairwood_sit",
    "Date07_Sub08_keyboard_move",
    "Date07_Sub08_keyboard_typing",
    "Date07_Sub08_monitor_hand",
    "Date07_Sub08_monitor_move",
    "Date07_Sub08_plasticcontainer",
    "Date07_Sub08_stool",
    "Date07_Sub08_suitcase",
    "Date07_Sub08_tablesmall",
    "Date07_Sub08_tablesquare",
    "Date07_Sub08_toolbox",
    "Date07_Sub08_trashbin",
    "Date07_Sub08_yogaball",
    "Date07_Sub08_yogamat"
  ],
  "test": [
    "Date03_Sub03_backpack_back",
    "Date03_Sub03_backpack_hand",
    "Date03_Sub03_backpack_hug",
    "Date03_Sub03_basketball",
    "Date03_Sub03_boxlarge",
    "Date03_Sub03_boxlong",
    "Date03_Sub03_boxmedium",
    "Date03_Sub03_boxsmall",
    "Date03_Sub03_boxtiny",
    "Date03_Sub03_chairblack_hand",
    "Date03_Sub03_chairblack_lift",
    "Date03_Sub03_chairblack_sit",
    "Date03_Sub03_chairblack_sitstand",
    "Date03_Sub03_chairwood_hand",
    "Date03_Sub03_chairwood_lift",
    "Date03_Sub03_chairwood_sit",
    "Date03_Sub03_keyboard_move",
    "Date03_Sub03_keyboard_typing",
    "Date03_Sub03_monitor_move",
    "Date03_Sub03_plasticcontainer",
    "Date03_Sub03_stool_lift",
    "Date03_Sub03_stool_sit",
    "Date03_Sub03_suitcase_lift",
    "Date03_Sub03_suitcase_move",
    "Date03_Sub03_tablesmall_lean",
    "Date03_Sub03_tablesmall_lift",
    "Date03_Sub03_tablesmall_move",
    "Date03_Sub03_tablesquare_lift",
    "Date03_Sub03_tablesquare_move",
    "Date03_Sub03_tablesquare_sit",
    "Date03_Sub03_toolbox",
    "Date03_Sub03_trashbin",
    "Date03_Sub03_yogaball_play",
    "Date03_Sub03_yogaball_sit",
    "Date03_Sub03_yogamat",
    "Date03_Sub04_backpack_back",
    "Date03_Sub04_backpack_hand",
    "Date03_Sub04_backpack_hug",
    "Date03_Sub04_basketball",
    "Date03_Sub04_boxlarge",
    "Date03_Sub04_boxlong",
    "Date03_Sub04_boxmedium",
    "Date03_Sub04_boxsmall",
    "Date03_Sub04_boxtiny",
    "Date03_Sub04_chairblack_hand",
    "Date03_Sub04_chairblack_liftreal",
    "Date03_Sub04_chairblack_sit",
    "Date03_Sub04_chairwood_hand",
    "Date03_Sub04_chairwood_lift",
    "Date03_Sub04_chairwood_sit",
    "Date03_Sub04_keyboard_move",
    "Date03_Sub04_keyboard_typing",
    "Date03_Sub04_monitor_hand",
    "Date03_Sub04_monitor_move",
    "Date03_Sub04_plasticcontainer_lift",
    "Date03_Sub04_stool_move",
    "Date03_Sub04_stool_sit",
    "Date03_Sub04_suitcase_ground",
    "Date03_Sub04_suitcase_lift",
    "Date03_Sub04_tablesmall_hand",
    "Date03_Sub04_tablesmall_lean",
    "Date03_Sub04_tablesmall_lift",
    "Date03_Sub04_tablesquare_hand",
    "Date03_Sub04_tablesquare_lift",
    "Date03_Sub04_tablesquare_sit",
    "Date03_Sub04_toolbox",
    "Date03_Sub04_trashbin",
    "Date03_Sub04_yogaball_play",
    "Date03_Sub04_yogaball_sit",
    "Date03_Sub04_yogamat",
    "Date03_Sub05_backpack",
    "Date03_Sub05_basketball",
    "Date03_Sub05_boxlarge",
    "Date03_Sub05_boxlong",
    "Date03_Sub05_boxmedium",
    "Date03_Sub05_boxsmall",
    "Date03_Sub05_boxtiny",
    "Date03_Sub05_chairblack",
    "Date03_Sub05_chairwood",
    "Date03_Sub05_keyboard",
    "Date03_Sub05_monitor",
    "Date03_Sub05_plasticcontainer",
    "Date03_Sub05_stool",
    "Date03_Sub05_suitcase",
    "Date03_Sub05_tablesmall",
    "Date03_Sub05_tablesquare",
    "Date03_Sub05_toolbox",
    "Date03_Sub05_trashbin",
    "Date03_Sub05_yogaball",
    "Date03_Sub05_yogamat"
  ]
}

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-seq',default="../../SHMR/data/BEHAVE/sequences/Date01_Sub01_backpack_back/")
    parser.add_argument('-config', default=None, help='if not given, load from info.json file')
    parser.add_argument('-o', '--out', default=None, help='if not given, save to the original sequence path')
    parser.add_argument('-fs', '--start', type=int, default=0)
    parser.add_argument('-fe', '--end', type=int, default=5)
    parser.add_argument('-t', '--target',default='person', choices=['person', 'obj'])
    parser.add_argument('-redo', default=False, action='store_true')
    #path=
    args = parser.parse_args()

    train_split=split_dict["train"]
    test_split=split_dict["test"]

    suffix_path="../../SHMR/data/BEHAVE/sequences/"

    trainGenerator=BEHAVE_Generator()
    all_len=0
    for file in train_split:
        args.seq=suffix_path+file

        args.end=len(get_all_file(args.seq, is_dir=True))
        main(args,trainGenerator)

    trainGenerator.save_data_list("../../SHMR/data/BEHAVE/processed/train_data.npy")

    testGenerator = BEHAVE_Generator()
    for file in test_split:
        args.seq = suffix_path + file
        args.end = len(get_all_file(args.seq, is_dir=True))
        main(args, testGenerator)

    testGenerator.save_data_list("../../SHMR/data/BEHAVE/processed/test_data.npy")