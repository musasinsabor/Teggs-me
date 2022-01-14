import unittest
from pathlib import Path
from model.dataset_creation import CocoTeggsme
from transformers import ViTFeatureExtractor
import numpy as np
import torch


class MyDataset(unittest.TestCase):
    """
    This tests makes three asserts:
        - The dataset size.
        - The train split size.
        - The test split size.
    """
    def test_dataset_size(self):
        my_img_folder = Path("C:/Users/Genesis/Teggs-me/dataset/images")
        my_ann_folder = Path("C:/Users/Genesis/Teggs-me/dataset/images/masks")
        my_ann_file = Path("C:/Users/Genesis/Teggs-me/dataset/coco_instances.json")
        my_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-large-patch32-384")
        self.dataset = CocoTeggsme(my_img_folder, my_ann_folder, my_ann_file, my_feature_extractor)
        np.random.seed(21)
        indices = np.random.randint(low=0, high=len(self.dataset), size=21)
        train_dataset = torch.utils.data.Subset(self.dataset, indices[6:])
        val_dataset = torch.utils.data.Subset(self.dataset, indices[:6])
        assert len(self.dataset) == 21
        assert len(train_dataset) == 15
        assert len(val_dataset) == 6
