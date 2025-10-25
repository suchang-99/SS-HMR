import hydra
import torch
from dataloader.Surreal_DataLoader import SurrealDepth
from dataloader.SLOPER4D_DataLoader import SLOPER4DDatset
from dataloader.BEHAVE_DataLoader import BEHAVEDataLoader
def get_dataset(cfg):

    #train_dataset=hydra.utils.instantiate(cfg["dataset"], _recursive_=False)
    print(cfg["dataset"])
    if(cfg["dataset"]["dataname"]=="surreal"):
        train_dataset=SurrealDepth(cfg["dataset"]["opt"])
        cfg["dataset"]["opt"]["isTrain"]=False
        test_dataset=SurrealDepth(cfg["dataset"]["opt"])
    elif (cfg["dataset"]["dataname"] == "sloper4d"):
        train_dataset = SLOPER4DDatset(cfg["dataset"]["opt"], mode="train")
        test_dataset = SLOPER4DDatset(cfg["dataset"]["opt"], mode="test")
    elif (cfg["dataset"]["dataname"] == "behave"):
        train_dataset = BEHAVEDataLoader(cfg["dataset"]["opt"], mode="train")
        test_dataset = BEHAVEDataLoader(cfg["dataset"]["opt"], mode="test")


    #test_dataset=hydra.utils.instantiate(cfg["dataset"], _recursive_=False)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=cfg["dataloader"]["train"]["batch_size"],
                                               shuffle=cfg["dataloader"]["train"]["shuffle"],
                                               num_workers=cfg["dataloader"]["train"]["num_workers"])
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=cfg["dataloader"]["test"]["batch_size"],
                                              shuffle=cfg["dataloader"]["test"]["shuffle"],
                                              num_workers=cfg["dataloader"]["test"]["num_workers"])
    return train_loader,train_dataset,test_loader,test_dataset

