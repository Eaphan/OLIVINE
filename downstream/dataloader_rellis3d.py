import os
import re
import torch
import numpy as np
from torch.utils.data import Dataset
from MinkowskiEngine.utils import sparse_quantize
from utils.transforms import make_transforms_clouds


TRAIN_SET = {0, 1, 2}
VALIDATION_SET = {3, 4}
TEST_SET = {}


def custom_collate_fn(list_data):
    """
    Collate function adapted for creating batches with MinkowskiEngine.
    """
    input = list(zip(*list_data))
    labelized = len(input) == 6
    if labelized:
        xyz, coords, feats, labels, evaluation_labels, inverse_indexes = input
    else:
        xyz, coords, feats, inverse_indexes = input

    coords_batch, len_batch = [], []

    for batch_id, coo in enumerate(coords):
        N = coords[batch_id].shape[0]
        coords_batch.append(
            torch.cat((torch.ones(N, 1, dtype=torch.int32) * batch_id, coo), 1)
        )
        len_batch.append(N)

    # Concatenate all lists
    coords_batch = torch.cat(coords_batch, 0).int()
    feats_batch = torch.cat(feats, 0).float()
    if labelized:
        labels_batch = torch.cat(labels, 0).long()
        return {
            "pc": xyz,  # point cloud
            "sinput_C": coords_batch,  # discrete coordinates (ME)
            "sinput_F": feats_batch,  # point features (N, 3)
            "len_batch": len_batch,  # length of each batch
            "labels": labels_batch,  # labels for each (voxelized) point
            "evaluation_labels": evaluation_labels,  # labels for each point
            "inverse_indexes": inverse_indexes,  # labels for each point
        }
    else:
        return {
            "pc": xyz,
            "sinput_C": coords_batch,
            "sinput_F": feats_batch,
            "len_batch": len_batch,
            "inverse_indexes": inverse_indexes,
        }


class Rellis3DDataset(Dataset):
    """
    Dataset returning a lidar scene and associated labels.
    Note that superpixels fonctionality have been removed.
    """

    def __init__(self, phase, config, transforms=None):
        self.phase = phase
        self.labels = self.phase != "test"
        self.transforms = transforms
        self.voxel_size = config["voxel_size"]
        self.cylinder = config["cylindrical_coordinates"]

        # a skip ratio can be used to reduce the dataset size
        # and accelerate experiments
        if phase == "train":
            try:
                skip_ratio = config["dataset_skip_step"]
            except KeyError:
                skip_ratio = 1
        else:
            skip_ratio = 1

        # official setting of rellis3d
        if phase in ("train", "parametrizing"):
            split_file = 'datasets/rellis3d/pt_train.lst'
        elif phase in ("val", "verifying"):
            split_file = 'datasets/rellis3d/pt_val.lst'
        elif phase == "test":
            split_file = 'datasets/rellis3d/pt_test.lst'

        self.list_files = ["datasets/rellis3d/" + item.strip().split()[0] for item in open(split_file, 'r').readlines()]

        # if phase in ("train", "parametrizing"):
        #     phase_set = TRAIN_SET
        # elif phase in ("val", "verifying"):
        #     phase_set = VALIDATION_SET
        # elif phase == "test":
        #     phase_set = TEST_SET

        # self.list_files = []
        # for num in phase_set:
        #     print('###', num)
        #     directory = next(
        #         os.walk(
        #             f"datasets/rellis3d/{num:0>5d}/os1_cloud_node_kitti_bin"
        #         )
        #     )
        #     self.list_files.extend(
        #         map(
        #             lambda x: f"datasets/rellis3d/"
        #             f"{num:0>5d}/os1_cloud_node_kitti_bin/" + x,
        #             directory[2],
        #         )
        #     )

        self.list_files = sorted(self.list_files)[::skip_ratio]
        # self.list_files = self.list_files[::10]
        #if phase in ("val", "verifying"):
        #    self.list_files = self.list_files[::10]

        # official github.com/unmannedlab/RELLIS-3D/blob/main/benchmarks/SalsaNext/train/tasks/semantic/config/labels/rellis.yaml
        # but we keep same setting with seal;
        learning_map = {
            0: 0, #"void"
            1: 1, #"dirt"
            3: 2, #"grass"
            4: 3, #"tree"
            5: 4, #"pole"
            6: 5, #"water"
            7: 6, #"sky"
            8: 7, #"vehicle"
            9: 8, #"object"
            10: 9, #"asphalt"
            12: 10, #"building"
            15: 11, #"log"
            17: 12, #"person"
            18: 13, #"fence"
            19: 14, #"bush"
            23: 15, #"concrete"
            27: 16, #"barrier"
            31: 17, #"puddle"
            33: 18, #"mud"
            34: 19, #"rubble"
        }
        self.eval_labels = np.zeros((np.max([k for k in learning_map.keys()]) + 1), dtype=np.int32)
        for k, v in learning_map.items():
            self.eval_labels[k] = v

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, idx):
        lidar_file = self.list_files[idx]
        # lidar_file = lidar_file.replace("os1", "vel")
        points = np.fromfile(lidar_file, dtype=np.float32).reshape((-1, 4))
        # get the points (4th coordinate is the point intensity)
        pc = points[:, :3]
        if self.labels:
            lidarseg_labels_filename = re.sub(
                "bin", "label", re.sub("_cloud_node_kitti_bin", "_cloud_node_semantickitti_label_id", lidar_file)
            )
            points_labels = (
                np.fromfile(lidarseg_labels_filename, dtype=np.uint32) & 0xFFFF
            )

        pc = torch.tensor(pc)

        # apply the transforms (augmentation)
        if self.transforms:
            pc = self.transforms(pc)

        if self.cylinder:
            # Transform to cylinder coordinate and scale for voxel size
            x, y, z = pc.T
            rho = torch.sqrt(x ** 2 + y ** 2) / self.voxel_size
            # corresponds to a split each 1°
            phi = torch.atan2(y, x) * 180 / np.pi
            z = z / self.voxel_size
            coords_aug = torch.cat((rho[:, None], phi[:, None], z[:, None]), 1)
        else:
            coords_aug = pc / self.voxel_size

        # Voxelization
        discrete_coords, indexes, inverse_indexes = sparse_quantize(
            coords_aug, return_index=True, return_inverse=True
        )
        unique_feats = torch.tensor(points[indexes][:, 3:] + 1.)

        if self.labels:
            points_labels = torch.tensor(
                np.vectorize(self.eval_labels.__getitem__)(points_labels),
                dtype=torch.int32,
            )
            unique_labels = points_labels[indexes]

        if self.labels:
            return (
                pc,
                discrete_coords,
                unique_feats,
                unique_labels,
                points_labels,
                inverse_indexes,
            )
        else:
            return pc, discrete_coords, unique_feats, inverse_indexes


def make_data_loader(config, phase, num_threads=0):
    """
    Create the data loader for a given phase and a number of threads.
    """
    # select the desired transformations
    if phase == "train":
        transforms = make_transforms_clouds(config)
    else:
        transforms = None

    # instantiate the dataset
    dset = Rellis3DDataset(phase=phase, transforms=transforms, config=config)
    collate_fn = custom_collate_fn
    batch_size = config["batch_size"] // config["num_gpus"]

    # create the loader
    loader = torch.utils.data.DataLoader(
        dset,
        batch_size=batch_size,
        # shuffle=False if sampler else True,
        shuffle=phase == "train",
        num_workers=num_threads,
        collate_fn=collate_fn,
        pin_memory=False,
        sampler=None,
        drop_last=False,
        worker_init_fn=lambda id: np.random.seed(torch.initial_seed() // 2 ** 32 + id),
    )
    return loader
