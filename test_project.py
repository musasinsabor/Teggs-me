import unittest
from pathlib import Path
from model.dataset_creation import CocoTeggsme
from model.pre_trained_model_prediction import PretrainedModelPrediction
from transformers import ViTFeatureExtractor
import numpy as np
import torch
from model.computer_vision_model import EggsViTModel
import os

directory = os.path.dirname(os.path.realpath(__file__))
my_data = "dataset"
my_imgs = "images"
my_masks = "masks"
my_file = "coco_instances.json"
prediction_file = "000001.png"
my_img_folder = os.path.join(directory, my_data, my_imgs)
my_ann_folder = os.path.join(directory, my_data, my_masks)
my_ann_file = os.path.join(directory, my_data, my_file)
my_pred_file = os.path.join(directory, my_data, my_imgs, prediction_file)


class MyDataset(unittest.TestCase):
    """
    This tests makes three asserts:
        - The dataset size.
        - The train split size.
        - The test split size.
    """

    def test_dataset_size(self):
        my_feature_extractor = ViTFeatureExtractor.from_pretrained("ydshieh/vit-gpt2-coco-en")
        self.dataset = CocoTeggsme(my_img_folder, my_ann_file)
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
        self.dataset = CocoTeggsme(my_img_folder, my_ann_file)
        my_parameters = {"model_name": 'facebook/detr-resnet-101-dc5',
                         "num_labels": 3,
                         "model_training_parameters": {"max_epochs": 1}}
        self.model = EggsViTModel(my_parameters, self.dataset)
        self.model.model_training()
        assert len(self.model.dataset) == 21


class MyPretrainedPrediction(unittest.TestCase):
    """
    My pretrained model prediction testing.
    """

    def test_pretrained_model_prediction(self):
        self.prediction = PretrainedModelPrediction('facebook/detr-resnet-101-dc5')
        pred = self.prediction.image_classification_classifier_pretrained(my_pred_file)
        assert pred != None

    def test_my_model_trained_prediction(self):
        self.prediction = PretrainedModelPrediction("./my_model")
        pred = self.prediction.image_classification_classifier_pretrained(
            my_pred_file)
        assert pred != None
