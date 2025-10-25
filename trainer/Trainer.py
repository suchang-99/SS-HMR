import time

import torch
import numpy as np

import utils.universal_utils as uni_utils
import time
import sys


class Trainer:
    def __init__(self, cfg,
                 model,
                 optimizer,
                 device,
                 optModel,
                 LossModel,
                 train_data_len,
                 ):
        self.cfg = cfg
        self.model = model
        self.optimizer = optimizer
        self.device = cfg["core"]["device"]
        self.optModel=optModel
        self.lossModel=LossModel

        self.selected_index = torch.tensor(np.load("data/preprocess_data/farthest_point_sample_index_0.npy", allow_pickle=True).item()[
            1024]).cuda()

        self.template=uni_utils.get_stand_pose_smpl()[0][:,self.selected_index]

        self.pred_segement=torch.zeros((train_data_len,1024),dtype=torch.long).cuda()
        self.pred_pose=torch.zeros((train_data_len,72)).cuda()
        self.pred_shape=torch.zeros((train_data_len,10)).cuda()
        self.pred_trans=torch.zeros((train_data_len,3)).cuda()
        self.pred_kp=torch.zeros((train_data_len,24,3)).cuda()

    def get_input_data(self,batch_data):

        data = batch_data["point_cloud"].clone()
        data = data.clone().cuda().to(torch.float32).squeeze()

        valid_point_mask = torch.sum(torch.abs(data), dim=2) > 1e-6

        data_offset = (torch.sum(data, dim=1) / valid_point_mask.sum(dim=1).view(-1, 1)).unsqueeze(1).to(torch.float32)
        data= data- data_offset
        data[~valid_point_mask]=0

        input_dict = {
            "batch_size": data.shape[0],
            "selected_index": self.selected_index,
            "dataset":self.cfg["dataset"]["dataname"],
            "source": {
                "pos": data,
                "valid_mask": valid_point_mask,
                "offset": data_offset,
            },
            "target":{
            "pos": self.template.clone().expand(data.shape[0],-1,-1).cuda(),
            },
        }
        if (self.cfg["dataset"]["dataname"] == "surreal" or self.cfg["dataset"]["dataname"] == "sloper4d"or self.cfg["dataset"]["dataname"] == "behave"):
            input_dict["recon"]= {
            "pose_param": batch_data["gt_pose"].clone().cuda(),
            "shape_param": batch_data["gt_shape"].clone().cuda(),
            "label_kp": batch_data["gt_joints3d"].clone().cuda() - data_offset }

            if(self.cfg["dataset"]["dataname"] == "behave"):
                input_dict["gender"]=batch_data["gender"]

            if(self.cfg["dataset"]["dataname"] == "sloper4d" or self.cfg["dataset"]["dataname"] == "behave"):
                input_dict["recon"]["trans_param"]=batch_data["camLoc"].clone().cuda()

        try:
            input_dict["recon"]["pred_segment"]=self.pred_segement[batch_data["index"]].clone().cuda()
        except Exception as e:
            print(e)
            pass
        return input_dict



    def train(self, epoch, data_loader,dataset):
        self.model.train()
        batch_len=len(data_loader)
        opt_gap=self.cfg["dataset"]["opt"]["opt_gap"]

        if(epoch%opt_gap==0):
            print("Using Opt Model")
            for i, batch in enumerate(data_loader):
                st=time.perf_counter()
                input_dict = self.get_input_data(batch)
                if(epoch==0):
                    new_mesh,new_j=self.optModel.direction_estimation(input_dict)

                else:
                    new_mesh,new_j= self.optModel.opt_direction_estimation(
                        input_dict,
                        self.pred_pose[batch["index"]].detach(),
                        self.pred_shape[batch["index"]].detach(),
                        self.pred_trans[batch["index"]].detach(),
                    )
                input_dict["recon"]["pred_mesh"]=new_mesh
                input_dict["recon"]["pred_kp"]=new_j
                corr_idx=uni_utils.get_corr_idx(input_dict["source"]["pos"],input_dict["recon"]["pred_mesh"][:,input_dict["selected_index"]])
                result_data=self.lossModel.calc_metrics(input_dict,is_test=True)

                result_data["time"]=torch.tensor(time.perf_counter()-st)
                self.lossModel.visualize_loss(epoch, i, batch_len, 0, result_data)
                self.pred_segement[batch["index"]]=corr_idx
                self.pred_kp[batch["index"]]=new_j
            torch.cuda.empty_cache()

            # save the results of optModel as the checkpoint
            # np.save(str(epoch)+"1pred_segement.npy",self.pred_segement.cpu().detach().numpy())
            # np.save(str(epoch)+"1pred_pose.npy", {
            #     "pose": self.pred_pose.cpu().detach().numpy(),
            #     "trans": self.pred_trans.cpu().detach().numpy(),
            #     "kp": self.pred_kp.cpu().detach().numpy(),
            # })
            self.lossModel.visual_epoch_info_and_save(epoch)

        for i, batch in enumerate(data_loader):
            input_dict = self.get_input_data(batch)
            result_dict = self.model(input_dict)

            loss = self.lossModel.calc_loss(epoch,i,batch_len,result_dict,self.pred_kp[batch["index"]])

            # update initialization
            if(epoch%opt_gap==(opt_gap-1)):
                self.pred_pose[batch["index"]]=result_dict["recon"]["pred_param"][:,:72]
                self.pred_shape[batch["index"]]=result_dict["recon"]["pred_param"][:,72:-3]
                self.pred_trans[batch["index"]]=result_dict["recon"]["pred_param"][:,-3:]
       
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        torch.cuda.empty_cache()

        self.lossModel.visual_epoch_info_and_save(epoch)


    def test(self, epoch, data_loader,dataset):
        self.model.eval()
        batch_len=len(data_loader)
        with torch.no_grad():
            for i, batch in enumerate(data_loader):
                input_dict = self.get_input_data(batch)
                result_dict = self.model(input_dict)
                loss = self.lossModel.calc_loss(epoch,i,batch_len,result_dict,1,is_test=True)
            self.lossModel.visual_epoch_info_and_save(epoch,self.model,is_test=True)



