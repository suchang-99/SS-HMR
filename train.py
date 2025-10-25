import hydra
from omegaconf import DictConfig, OmegaConf
import platform
from dataloader.get_dataset import get_dataset
import numpy as np
import torch

from model.OptModel import OptModel
from trainer.Trainer import Trainer
from model.HMRModel import HMRModel
import torch.optim as optim
from loss.loss import LossModel
import random



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(cfg: DictConfig) -> None:
    print(cfg["core"])
    setup_seed(6)
    train_datalodar,train_dataset,test_dataloader,test_dataset=get_dataset(cfg)

    device="cuda:0"
    optModel = OptModel()

    hmrModel=HMRModel(cfg).to(device)

    lossModel=LossModel(cfg)

    optimizer = optim.Adam(hmrModel.parameters(), lr=0.0005)

    train_data_len=len(train_dataset)
    myTrainer=Trainer(  cfg=cfg,
                        model=hmrModel,
                        optimizer=optimizer,
                        device=device,
                        optModel=optModel,
                        LossModel=lossModel,
                        train_data_len=train_data_len,
                        )

    for epoch in range(1000):
        myTrainer.train(epoch,train_datalodar,train_dataset)
        myTrainer.test(epoch,test_dataloader,test_dataset)


@hydra.main(version_base=None, config_path="configs", config_name="config")
def my_app(cfg : DictConfig) -> None:
    if(platform.system()=="Windows"):
        cfg.core.is_debug = True
    train(cfg)



if __name__ == "__main__":
    my_app()