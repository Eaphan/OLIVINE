import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from utils.transforms import make_transforms_clouds
from downstream.dataloader_kitti import SemanticKITTIDataset
from downstream.dataloader_scribble_kitti import ScribbleKITTIDataset
from downstream.dataloader_rellis3d import Rellis3DDataset
# from downstream.dataloader_semanticposs import SemanticPOSSDataset
from downstream.dataloader_nuscenes import NuScenesDataset, custom_collate_fn
from downstream.dataloader_semanticstf import SemanticSTFDataset
# from downstream.dataloader_synlidar import SynLiDARDataset
from downstream.dataloader_daps3d import DAPS3DDataset

class DownstreamDataModule(pl.LightningDataModule):
    """
    The equivalent of a DataLoader for pytorch lightning.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        # in multi-GPU the actual batch size is that
        self.batch_size = config["batch_size"] // config["num_gpus"]
        # the CPU workers are split across GPU
        self.num_workers = max(config["num_threads"] // config["num_gpus"], 1)

    def setup(self, stage):
        # setup the dataloader: this function is automatically called by lightning
        transforms = make_transforms_clouds(self.config)
        if self.config["dataset"].lower() == "nuscenes":
            Dataset = NuScenesDataset
        elif self.config["dataset"].lower() in ("kitti", "semantickitti"):
            Dataset = SemanticKITTIDataset
        elif self.config["dataset"].lower() in ("scribblekitti", "scribble_kitti"):
            Dataset = ScribbleKITTIDataset
        elif self.config["dataset"].lower() == "rellis3d":
            Dataset = Rellis3DDataset            
        # elif self.config["dataset"].lower() in ("semanticposs", "semantic_poss"):
        #     Dataset = SemanticPOSSDataset
        elif self.config["dataset"].lower() in ("semanticstf", "semantic_stf"):
            Dataset = SemanticSTFDataset        
        # elif self.config["dataset"].lower() in ("synlidar"):
        #     Dataset = SynLiDARDataset
        elif self.config["dataset"].lower() in ("daps3d"):
            Dataset = DAPS3DDataset   
        else:
            raise Exception(f"Unknown dataset {self.config['dataset']}")
        if self.config["training"] in ("parametrize", "parametrizing"):
            phase_train = "parametrizing"
            phase_val = "verifying"
        else:
            phase_train = "train"
            phase_val = "val"
        self.train_dataset = Dataset(
            phase=phase_train, transforms=transforms, config=self.config
        )
        if Dataset == NuScenesDataset:
            self.val_dataset = Dataset(
                phase=phase_val,
                config=self.config,
                cached_nuscenes=self.train_dataset.nusc,
            )
        else:
            self.val_dataset = Dataset(phase=phase_val, config=self.config)
        print("### len of train_dataset", len(self.train_dataset))
        print("### len of val_dataset", len(self.val_dataset))

    def train_dataloader(self):
        # construct the training dataloader: this function is automatically called
        # by lightning
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            drop_last=True,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )

    def val_dataloader(self):
        # construct the validation dataloader: this function is automatically called
        # by lightning
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=custom_collate_fn,
            pin_memory=True,
            drop_last=False,
            worker_init_fn=lambda id: np.random.seed(
                torch.initial_seed() // 2 ** 32 + id
            ),
        )
