"""

Modified from VoteHMR
===========================================================

**Original Project**: VoteHMR: Occlusion-Aware Voting Network for Robust 3D Human Mesh Recovery from Partial Point Clouds
**Source URL**: https://github.com/hanabi7/VoteHMR

"""
import copy
import sys
import os.path as osp
import os
import time

import numpy as np
import torch
import random
import pickle
import torch.utils.data as data
from huggingface_hub import upload_file
from scipy.io import loadmat
import math
import transforms3d
# import utils.wod_utils as wod_utils

cur_path = os.path.abspath(os.path.dirname(__file__))
root_path = os.path.split(cur_path)[0]
root_path = os.path.split(root_path)[0]
sys.path.append(root_path)


class PointsSample():
    def __init__(self, opt):
        self.number_points_sample = opt.number_points_sample
        self.noise_mean = opt.noise_mean
        self.noise_var = opt.noise_var

    def get_intrinsic(self):
        # These are set in Blender (datageneration/main_part1.py)
        res_x_px = 320  # *scn.render.resolution_x
        res_y_px = 240  # *scn.render.resolution_y
        f_mm = 60  # *cam_ob.data.lens
        sensor_w_mm = 32  # *cam_ob.data.sensor_width
        sensor_h_mm = sensor_w_mm * res_y_px / res_x_px  # *cam_ob.data.sensor_height (function of others)

        scale = 1  # *scn.render.resolution_percentage/100
        skew = 0  # only use rectangular pixels
        pixel_aspect_ratio = 1

        # From similar triangles:
        # sensor_width_in_mm / resolution_x_inx_pix = focal_length_x_in_mm / focal_length_x_in_pix
        fx_px = f_mm * res_x_px * scale / sensor_w_mm
        fy_px = f_mm * res_y_px * scale * pixel_aspect_ratio / sensor_h_mm

        # Center of the image
        u = res_x_px * scale / 2
        v = res_y_px * scale / 2

        # Intrinsic camera matrix
        K = np.array([[fx_px, skew, u], [0, fy_px, v], [0, 0, 1]])
        return K

    def get_extrinsic(self, T):
        # Take the first 3 columns of the matrix_world in Blender and transpose.
        # This is hard-coded since all images in SURREAL use the same.
        R_world2bcam = np.array([[0, 0, 1], [0, -1, 0], [-1, 0, 0]]).transpose()
        # *cam_ob.matrix_world = Matrix(((0., 0., 1, params['camera_distance']),
        #                               (0., -1, 0., -1.0),
        #                               (-1., 0., 0., 0.),
        #                               (0.0, 0.0, 0.0, 1.0)))

        # Convert camera location to translation vector used in coordinate changes
        T_world2bcam = -1 * np.dot(R_world2bcam, T)

        # Following is needed to convert Blender camera to computer vision camera
        R_bcam2cv = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])

        # Build the coordinate transform matrix from world to computer vision camera
        R_world2cv = np.dot(R_bcam2cv, R_world2bcam)
        T_world2cv = np.dot(R_bcam2cv, T_world2bcam)

        # Put into 3x4 matrix
        RT = np.concatenate([R_world2cv, T_world2cv], axis=1)
        return RT, R_world2cv, T_world2cv

    def farthest_point_sample(self, xyz, npoint):
        """
        Input:
            xyz: pointcloud data, [N, C]
            npoint: number of samples
        Return:
            centroids: sampled pointcloud index, [npoint]
        """
        device = xyz.device
        N, C = xyz.shape
        centroids = torch.zeros(npoint, dtype=torch.long).to(device)
        distance = torch.ones(N).to(device) * 1e10
        farthest = torch.randint(0, N, (1,), dtype=torch.long).to(device)
        for i in range(npoint):
            centroids[i] = farthest
            centroid = xyz[farthest, :].view(1, 3)
            dist = torch.sum((xyz - centroid) ** 2, -1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, -1)[1]
        return centroids

    def point_clouds_sampler(self, point_clouds, segment):
        """
        :param point_clouds: [number_points, 3] tensor
        :param segment: [number_points]  tensor
        :param n_sample:
        :return:
        """
        num_availabel = point_clouds.shape[0]
        if num_availabel >= self.number_points_sample:
            """centroids = self.farthest_point_sample(point_clouds, n_sample)
            # centroids is a tensor of size [n_points]
            centroids = centroids.long()"""
            sample_index = random.sample(range(0, num_availabel), self.number_points_sample)
            sample_index = np.array(sample_index)
            #print(sample_index.shape)
            #sample_index = np.linspace(0, num_availabel-1, self.number_points_sample).astype(int).flatten()
            #print(sample_index.shape)

            sample_index = torch.from_numpy(sample_index).long()
            sample_point_clouds = point_clouds[sample_index, :]
            sample_segment = segment[sample_index]
        else:
            num_offset = self.number_points_sample - num_availabel
            rand_inds = np.random.randint(0, high=num_availabel, size=num_offset)
            rand_inds = torch.from_numpy(rand_inds).long()
            sample_point_clouds = point_clouds[rand_inds, :]
            sample_segment = segment[rand_inds]
            sample_point_clouds = torch.cat((point_clouds, sample_point_clouds), dim=0)
            sample_segment = torch.cat((segment, sample_segment), dim=0)
        return sample_point_clouds, sample_segment

    def gaussion_random_generator(self, point_clouds):
        noise = np.random.normal(self.noise_mean, self.noise_var, point_clouds.shape)
        noise = torch.from_numpy(noise)
        out = point_clouds + noise
        return out

    def point_cloud_generate(self, depth_image, seg_image, camLoc):
        """
        Inputs
            depth_image: [240, 320]
            seg_image:   [240, 320]
            camDist:     [3]
        Return:
            Pointclouds: [number_sample, 3]
            Gt_segment: [number_sample]
        """
        width, height = depth_image.shape
        # width 240, height 320
        seg_mask = np.where(seg_image != 0)
        y = seg_mask[0]
        x = seg_mask[1]
        number_points = x.shape[0]
        z = depth_image[y, x]
        x = x.reshape(number_points, 1)
        y = y.reshape(number_points, 1)
        z = z.reshape(number_points, 1)
        x = (x - 160) * z / 600
        y = (y - 120) * z / 600
        cam_coordinates = np.concatenate((x, y, z), axis=1)
        wrd_x = - cam_coordinates[:, 2] + camLoc[0]
        wrd_y = cam_coordinates[:, 1] + camLoc[1]
        wrd_z = - cam_coordinates[:, 0] + camLoc[2]
        wrd_x = wrd_x.reshape(number_points, 1)
        wrd_y = wrd_y.reshape(number_points, 1)
        wrd_z = wrd_z.reshape(number_points, 1)
        point_clouds = np.concatenate((wrd_x, wrd_y, wrd_z), axis=1)
        gt_segment = seg_image[seg_mask[0], seg_mask[1]]
        gt_segment = gt_segment.reshape(number_points)
        gt_segment = torch.from_numpy(gt_segment)
        point_clouds = torch.from_numpy(point_clouds)
        if number_points > 0:
            #print(point_clouds.shape)
            point_clouds, gt_segment = self.point_clouds_sampler(point_clouds, gt_segment)
            return point_clouds, gt_segment
        else:
            return point_clouds, gt_segment

def update_data_dir(item, replace_from, replace_to):
    """
    更新 item 中的 'data_dir' 键，将 replace_from 替换为 replace_to。
    """
    item["data_dir"] = item["data_dir"].replace(replace_from, replace_to)
    return item

class SurrealDepth(data.Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.isTrain = opt.isTrain
        self.point_sampler = PointsSample(opt)
        self.data_list = []
        self.use_generated_data_file = opt.use_generated_data_file
        self.surreal_save_path = opt.surreal_save_path
        self.my_index=0

        if self.isTrain:
            self.split = 'train'
        else:
            self.split = 'test'

        self.surreal_split_name = ['run0', 'run1', 'run2']
        self.surreal_train_sequence_number = opt.surreal_train_sequence_number
        self.surreal_train_sequence_sample_number = opt.surreal_train_sequence_sample_number
        self.surreal_test_sequence_number = opt.surreal_test_sequence_number
        self.surreal_test_sequence_sample_number = opt.surreal_test_sequence_sample_number
        self.data_path = opt.surreal_dataset_path + self.split

        if (not self.use_generated_data_file or os.path.exists(self.surreal_save_path + self.split + '_annotation.npy')):
            cmu_keys, cmu_keys_name = self.load_surreal_data(self.data_path)
            print('the lens of cmu_keys:', len(cmu_keys))
            complete_sequence = self.sequence_complete(cmu_keys, cmu_keys_name)
            random.shuffle(complete_sequence)
            if self.split == 'train':
                sequence = complete_sequence[:self.surreal_train_sequence_number]
            else:
                sequence = complete_sequence[:self.surreal_test_sequence_number]
            number_sequence = len(sequence)
            for i in range(number_sequence):
                self.single_sequence_process(sequence[i], i)
            self.generate_datasets()
        else:
            if self.split == 'train':
                file_name = self.surreal_save_path + self.split + '_annotation.npy'
                self.data_list = np.load(file_name, allow_pickle=True)
             
            else:
                file_name = self.surreal_save_path + self.split + '_annotation.npy'
                self.data_list = np.load(file_name, allow_pickle=True)
        self.data_list=self.data_list
        # for tmp_i in range(0,len(self.data_list)):
        #     self.data_list[tmp_i]["data_dir"]=self.data_list[tmp_i]["data_dir"].replace("data/",opt.data_dir_replace)


    def rotateBody(self, RzBody, pelvisRotVec):
        angle = np.linalg.norm(pelvisRotVec)
        Rpelvis = transforms3d.axangles.axangle2mat(pelvisRotVec / angle, angle)
        globRotMat = np.dot(RzBody, Rpelvis)
        R90 = transforms3d.euler.euler2mat(np.pi / 2, 0, 0)
        globRotAx, globRotAngle = transforms3d.axangles.mat2axangle(np.dot(R90, globRotMat))
        globRotVec = globRotAx * globRotAngle
        return globRotVec

    def sequence_complete(self, cmu_keys, cmu_keys_name):
        length = len(cmu_keys)
        complete_sequence = []
        for i in range(length):
            sequence_path = cmu_keys[i]
            sequence_name = cmu_keys_name[i]
            number_files = len(os.listdir(sequence_path))
            number_sequence = int(number_files / 4)
            for j in range(number_sequence):
                depth_filename = sequence_name + '_c00%02d' % (j + 1) + '_depth.mat'
                segm_filename = sequence_name + '_c00%02d' % (j + 1) + '_segm.mat'
                info_filename = sequence_name + '_c00%02d' % (j + 1) + '_info.mat'
                depth_filepath = sequence_path + '/' + depth_filename
                segm_filepath = sequence_path + '/' +segm_filename
                info_filepath = sequence_path + '/' +info_filename
                if os.path.exists(depth_filepath):
                    file_dict = dict(
                        depth=depth_filepath,
                        segm=segm_filepath,
                        info=info_filepath,
                        sequence_name=sequence_name,
                        index=j
                    )
                    complete_sequence.append(file_dict)
        print('sequence_lenghth:', len(complete_sequence))
        return complete_sequence

    def data_generate(self, depth_filename, info_filename, segm_filename, sample_number, index):
        depth_file = loadmat(depth_filename)
        info_file = loadmat(info_filename)
        segm_file = loadmat(segm_filename)
        pose = info_file['pose']
        # [10, 100]
        shape = info_file['shape']
        # [72, 100]
        number_frame = pose.shape[1]
        joints3d = info_file['joints3D']
        # joints3d [3, 24, 100]
        camdist = info_file['camDist']
        camLoc = info_file['camLoc']
        gt_joints2d = info_file['joints2D']
        gender = info_file['gender']
        zrot = info_file['zrot']
        zrot = zrot[0][0]
        RzBody = np.array(((math.cos(zrot), -math.sin(zrot), 0),
                           (math.sin(zrot), math.cos(zrot), 0),
                           (0, 0, 1)))
        sequence_path = self.surreal_save_path + self.split + '/' + str(index) + '/'
        if not osp.exists(sequence_path):
            os.mkdir(sequence_path)
        # gender is a numpy array of size [number_frame, 1]
        if joints3d.ndim == 3:
            for i in range(min(sample_number, number_frame)):
                idx = i
                depth_image = depth_file['depth_%d' % (idx + 1)]
                segm_image = segm_file['segm_%d' % (idx + 1)]
                center = gt_joints2d[:, 0, idx]
                gt_joints3d = torch.from_numpy(joints3d[:, :, idx]).transpose(1, 0)
                pose_param = pose[:, idx]
                pose_param[0:3] = self.rotateBody(RzBody, pose_param[0:3])
                pose_param = torch.from_numpy(pose_param)
                shape_param = torch.from_numpy(shape[:, idx])

                data_file_dir = sequence_path + str(idx) + '_dict.pkl'
                point_cloud, segmentation = self.point_sampler.point_cloud_generate(depth_image, segm_image, camLoc)
                # point_cloud = self.point_sampler.gaussion_random_generator(point_cloud)
                if point_cloud.shape[0] == 1024:
                    segmentation = segmentation - 1
                    num_segment = len(np.unique(segmentation))

                    gt_joints3d[:,1]=gt_joints3d[:,1]*-1
                    point_cloud[:,1]=point_cloud[:,1]*-1

                    if num_segment > 18:
                        single_data = dict(
                            data_dir=data_file_dir,
                            gt_joints3d=gt_joints3d,
                            gt_pose=pose_param,
                            gt_shape=shape_param,
                            camLoc=camLoc
                        )
                        croped_image = self.crop_single_image(depth_image, center)

                        data_dict = {
                            'point_cloud':point_cloud,
                            'gt_segment':segmentation
                        }
                        with open(data_file_dir, 'wb') as f:
                            pickle.dump(data_dict, f)
                        self.data_list.append(single_data)
                else:
                    print(point_cloud.shape)
            print('sequence_complete')

    def crop_single_image(self, depth_image, center, crop_size=224):
        h, w = depth_image.shape
        left = np.maximum(center[0] - 0.5 * crop_size, 0)
        top = np.maximum(center[1] - 0.5 * crop_size, 0)
        right = left + crop_size
        down = top + crop_size
        right = np.minimum(right, w)
        down = np.minimum(down, h)
        left = right - crop_size
        top = down - crop_size
        top = top.astype(np.int32)
        down = down.astype(np.int32)
        left = left.astype(np.int32)
        right = right.astype(np.int32)
        croped_image = depth_image[top:down, left:right]
        return croped_image

    def single_sequence_process(self, sequence_path, i):
        """
        :param data_path: the data_path for the small sequence
        :return:
        """
        depth_path = sequence_path['depth']
        segm_path = sequence_path['segm']
        info_path = sequence_path['info']
        index = i
        if self.split == 'train':
            self.data_generate(depth_path, info_path, segm_path, self.surreal_train_sequence_sample_number, index)
        else:
            self.data_generate(depth_path, info_path, segm_path, self.surreal_test_sequence_sample_number, index)

    def load_surreal_data(self, path):
        """
        Input:
            path: /data/liuguanze/datasets/surreal/SURREAL/data/cmu/spilt
        Return:
            cmu_keys [list of sequences path]
        """
        cmu_keys = []
        cmu_keys_name = []
        for filename in self.surreal_split_name:
            added_path = path + '/' + filename + '/'
            for sequence_name in os.listdir(added_path):
                # print('the content of the sequence_path:', sequence_path)
                sequence_path = added_path + sequence_name
                if os.path.isdir(sequence_path):
                    cmu_keys_name.append(sequence_name)
                    cmu_keys.append(sequence_path)
        return cmu_keys, cmu_keys_name
        # number of sequences of the split of the surreal datasets

    def __getitem__(self, index):

        single_data = self.data_list[index]

        data_file = single_data['data_dir']
        with open(data_file, 'rb') as f:
            data_dict = pickle.load(f, encoding='bytes')
        single_data['point_cloud'] = data_dict['point_cloud']

        single_data['gt_segment'] = data_dict['gt_segment']
        single_data['index']=index

        return single_data

    def __len__(self):
        return len(self.data_list)

    def generate_datasets(self):
        file_name = self.surreal_save_path + self.split + '_annotation.npy'
        np.save(file_name, self.data_list)
