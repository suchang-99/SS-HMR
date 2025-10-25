import os
import argparse

import pickle
import torch
import numpy as np
import open3d as o3d
from torch.utils.data import Dataset, DataLoader



class SLOPER4DDatset(Dataset):

    def __init__(self, cfg, mode):

        self.cfg = cfg
        self.mode = mode

        if(mode=="train"):
            self.data_list = np.load("data/SLOPER4D/ntrain_data.npy", allow_pickle=True)
        else:
            self.data_list = np.load("data/SLOPER4D/ntest_data.npy", allow_pickle=True)
        if(mode!="train"):
            tmp_list=[]
            for item in self.data_list:
                tdata=torch.tensor(item["point_cloud"])

                mask= torch.sum(torch.abs(tdata), dim=1) > 1e-6
                if(torch.sum(mask)<512):
                    tmp_list.append(item)

        self.len=len(self.data_list)
        print(len(self.data_list))

    def __len__(self):
        return self.len

    def __getitem__(self, index):

        single_data = self.data_list[index]
        single_data['index']=index

        return single_data