import unittest
from pathlib import Path
from model.dataset_creation import CocoTeggsme
from transformers import ViTFeatureExtractor
import numpy as np
import torch
from model.computer_vision_model import EggsViTModel



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
        my_feature_extractor = ViTFeatureExtractor.from_pretrained("ydshieh/vit-gpt2-coco-en")
        self.dataset = CocoTeggsme("C:/Users/Genesis/Teggs-me/dataset/images", my_ann_file)
        np.random.seed(21)
        indices = np.random.randint(low=0, high=len(self.dataset), size=21)
        train_dataset = torch.utils.data.Subset(self.dataset, indices[6:])
        val_dataset = torch.utils.data.Subset(self.dataset, indices[:6])
        assert len(self.dataset) == 21
        assert len(train_dataset) == 15
        assert len(val_dataset) == 6


class MyModel(unittest.TestCase):
    """
    My model testing.
    """

    def test_model(self):
        self.dataset = CocoTeggsme("C:/Users/Genesis/Teggs-me/dataset/images",
                                   "C:/Users/Genesis/Teggs-me/dataset/_annotations.coco.json")
        my_parameters = {"model_name": 'facebook/detr-resnet-101-panoptic',
                         "num_labels": 3,
                         "model_training_parameters": {"max_epochs": 1}}
        self.model = EggsViTModel(my_parameters, self.dataset)
        self.model.model_training()
        assert len(self.model.dataset) == 21
