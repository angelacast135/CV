
# Standard library imports
import os
import re
import os.path as osp

# Numpy/Scipy imports
import numpy as np
import scipy.io as sio

# Torch imports
import torch
from torch.utils import data

# Other imports
import cv2
import tqdm
import pdb

class LiverException(Exception):
    pass


class LiverDataset(data.Dataset):
    """docstring for LiverLoader"""
    SPLIT_FOLDER_NAME = 'FCN_{0}'
    FOLDER_RE = re.compile(r'^vol_(\d{1,3})$')
    SPLITS = ['train', 'test', 'val']
    LABELS = {'liver': 1, 'tumor': 2}

    def __init__(self, data_root, split, data_folder="data", transform=None, 
                 annotation_transform=None):
        super(LiverDataset, self).__init__()
        self.data_root = data_root
        self.data_folder = data_folder
        self.split = split
        self.dataset = []
        self.transform = transform
        self.annotation_transform = annotation_transform
        self.split_file = osp.join(self.data_folder, self.split) + '.pth' 

        if split not in self.SPLITS:
            raise LiverException(
                "Split {0} not in {1}".format(split, self.SPLITS))

        if not osp.exists(self.data_folder):
            self.process_dataset()

        self.dataset = torch.load(self.split_file)

    def process_dataset(self):
        os.makedirs(self.data_folder)
        print("Generating splits from {0}".format(self.data_root))
        for split in self.SPLITS:
            split_ids = {'labels':[], 'images':[]}
            print("Generating {0} split".format(split))
            split_folder = osp.join(self.data_root,
                                    self.SPLIT_FOLDER_NAME.format(split))
            print("Loading folder {0}".format(split_folder))
            for root, dirs, files in tqdm.tqdm(os.walk(split_folder)):
                match = self.FOLDER_RE.match(osp.basename(root))  
                if match is not None:
                    # print(root)
                    num_ann = match.groups(0)[0]
                    files = [x for x in files if osp.splitext(x)[1] == '.mat']
                    files = sorted(files)
                    for filename, var_name in zip(
                        files, ['anot_test', 'vol_test']):
                        if osp.splitext(filename)[1] == '.mat':
                            filename = osp.join(root, filename)
                            mat = sio.loadmat(filename)[var_name]
                            mat += np.abs(mat.min())
                            mat = mat / mat.max()
                            for idx in range(0, mat.shape[-1]):
                                slice_ = mat[..., idx]
                                # print(slice_.shape)
                                split_key = 'images'
                                slice_name = '{0}_{1}_{2}.{3}'# .format(
                                #    num_ann, idx, '.pth')
                                if var_name == 'anot_test':
                                    split_key = 'labels'
                                    # slice_name = slice_name.format(num_ann, idx, split_key[:-1], 'pth')
                                    slice_ = np.stack([
                                        (slice_ == self.LABELS[x]).astype(
                                            np.float64) for x in self.LABELS])
                                slice_name = slice_name.format(num_ann, idx, split_key[:-1], 'pth')    
                                torch.save(torch.from_numpy(slice_), osp.join(root, slice_name))
                                split_ids[split_key].append(
                                    osp.join(root, slice_name))
                                # print(osp.join(root, slice_name))
            # print(split_ids)
            torch.save(
                list(zip(split_ids['images'], split_ids['labels'])),
                osp.join(self.data_folder, split) + '.pth')


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_file, ann_file = self.dataset[idx]
        image = torch.load(image_file).float()
        image = image.repeat(3,1,1).squeeze()
        # print(torch.is_tensor(image))
        target = torch.load(ann_file).long()
        target = target[0] * 1 + target[1] * 2

        # image = image.unsqueeze(0)
        if self.transform is not None:
            image = self.transform(image)
        if self.annotation_transform is not None:
            target = self.annotation_transform(target)
        return image, target

    def untransform(self, img, lbl):
        img = img.numpy()
        lbl = lbl.numpy()
        return img, lbl

