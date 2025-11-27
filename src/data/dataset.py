import os
import torch
import rasterio
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class Sen1Floods11Dataset(Dataset):
    """
    Dataset class for Sen1Floods11 using HandLabeled data.
    Loads S1 (SAR), S2 (Optical), and Label (Water Mask).
    """
    def __init__(self, root_dir, split="train", transform=None):
        """
        Args:
            root_dir (str): Path to root data directory (e.g., 'data/sen1floods11')
            split (str): 'train' or 'test' (loads from generated csv splits)
            transform (callable, optional): Albumentations transforms
        """
        self.root_dir = root_dir
        self.transform = transform
        self.split = split
        
        # paths assuming standard Sen1Floods11 structure
        # v1.1/data/flood_events/HandLabeled/
        self.base_path = os.path.join(root_dir, "v1.1", "data", "flood_events", "HandLabeled")
        self.s1_dir = os.path.join(self.base_path, "S1Hand")
        self.s2_dir = os.path.join(self.base_path, "S2Hand")
        self.label_dir = os.path.join(self.base_path, "LabelHand")
        
        # load file IDs from CSV splits if they exist
        split_dir = os.path.join(root_dir, "splits")
        csv_path = os.path.join(split_dir, f"{split}_split.csv")
        
        if os.path.exists(csv_path):
            print(f"Loading {split} split from {csv_path}")
            df = pd.read_csv(csv_path)
            # use the 'id' column which contains filenames like 'Region_ID_LabelHand'
            self.file_ids = df["id"].tolist()
        else:
            print(f"Warning: Split file {csv_path} not found.")
            print("Falling back to scanning directory (ignoring splits)...")
            if os.path.exists(self.label_dir):
                all_files = sorted([f.replace(".tif", "") for f in os.listdir(self.label_dir) if f.endswith(".tif")])
                # fallback: 80/20 split if no CSV found
                split_idx = int(0.8 * len(all_files))
                if split == "train":
                    self.file_ids = all_files[:split_idx]
                else:
                    self.file_ids = all_files[split_idx:]
            else:
                self.file_ids = []

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        
        # constructing paths
        # Sen1Floods11 filenames usually look like: 'Bolivia_428833_S1Hand.tif'
        # the ID here would be 'Bolivia_428833_LabelHand' if we just stripped .tif
        # we need to reconstruct the S1 and S2 filenames 
        
        # extract region_id from "Region_ID_LabelHand"
        # example: Bolivia_428833_LabelHand -> Bolivia_428833
        region_id = file_id.replace("_LabelHand", "")
        
        s1_path = os.path.join(self.s1_dir, f"{region_id}_S1Hand.tif")
        s2_path = os.path.join(self.s2_dir, f"{region_id}_S2Hand.tif")
        label_path = os.path.join(self.label_dir, f"{file_id}.tif")
        
        # load data
        s1 = self._load_tif(s1_path)      # Shape: (C, H, W)
        s2 = self._load_tif(s2_path)      # Shape: (C, H, W)
        label = self._load_tif(label_path) # Shape: (1, H, W) usually
        
        # preprocessing / normalization
        s1 = self._preprocess_s1(s1)
        s2 = self._preprocess_s2(s2)
        
        # prepare dictionary for transforms
        # albumentations expects HWC numpy arrays
        data = {
            "image": np.moveaxis(s2, 0, -1),   # treat S2 as main image
            "s1": np.moveaxis(s1, 0, -1),      # pass S1 as extra
            "mask": label.squeeze()            # (H, W)
        }
        
        if self.transform:
            augmented = self.transform(**data)
            s2 = augmented["image"]
            s1 = augmented["s1"]
            label = augmented["mask"]
            
            # convert back to CHW tensor
            s2 = torch.from_numpy(np.moveaxis(s2, -1, 0)).float()
            s1 = torch.from_numpy(np.moveaxis(s1, -1, 0)).float()
            label = torch.from_numpy(label).long()
        else:
            # basic ToTensor conversion
            s2 = torch.from_numpy(s2).float()
            s1 = torch.from_numpy(s1).float()
            label = torch.from_numpy(label).long()

        return {
            "s1": s1, 
            "s2": s2, 
            "label": label,
            "id": region_id
        }

    def _load_tif(self, path):
        """Helper to load TIF using rasterio"""
        if not os.path.exists(path):
            # fallback for missing modalities (if only S1 exists)
            # raise error to be safe
            raise FileNotFoundError(f"File not found: {path}")
            
        with rasterio.open(path) as src:
            image = src.read() # (C, H, W)
        return image.astype(np.float32)

    def _preprocess_s1(self, img):
        """
        Clip and Normalize Sentinel-1 Data.
        Raw S1 is usually in dB or linear intensity.
        Sen1Floods11 S1Hand is usually IW mode (VV, VH).
        Common practice: Clip to [-25, 0] dB and scale to [0, 1].
        """
        # assume input is already dB. If linear, need 10*log10(x).
        # Sen1Floods11 'S1Hand' is usually processed
        # lets assume standard clipping for now
        img = np.clip(img, -25, 0)
        img = (img + 25) / 25 # Scale to [0, 1]
        return img

    def _preprocess_s2(self, img):
        """
        Normalize Sentinel-2 Data.
        Raw S2 is usually scaled by 10000.
        """
        img = img / 10000.0
        img = np.clip(img, 0, 1)
        return img
