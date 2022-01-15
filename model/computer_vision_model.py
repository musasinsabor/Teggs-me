from transformers import ViTFeatureExtractor, ViTForImageClassification, ViTConfig
from typing import Dict
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
import numpy as np
import torch
from torch.utils.data import DataLoader


class EggsViTModel:
    """This is a computer vision model in ImageClassification task wrapped in huggingface."""

    def __init__(self,
                 parameters,
                 dataset):
        """It takes a parameters Dict where it can be specified the following data:
            model_name(str): Model name in huggingface/models storage.
            num_labels(int): Specification of the label numbers.
            model_training_parameters(Dict): This Dict can include the following parameters:
                lr,
                fp16(bool)
        """
        self.parameters: Dict = parameters
        self.model_name = parameters["model_name"]
        self.num_labels = parameters["num_labels"]
        self.model_training_parameters = parameters["model_training_parameters"]
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name)
        self.dataset = self.dataset_obj(dataset)
        self.model = self.pretrained_model()
        self.my_dataset = dataset

    def dataset_obj(self, dataset):
        """
        This method uses the dataset info to create a dataset instance.
        :returns a Dataset object
        """
        return dataset

    def pretrained_model(self):
        """
        This method specify a pretrained ViModel.
        """
        config = ViTConfig.from_pretrained(self.model_name, ignore_mismatched_sizes=True)
        config.num_labels = self.num_labels
        pt_model = ViTForImageClassification(config)
        return pt_model

    def model_training(self):
        # let's split it up into very tiny training and validation sets using random indices
        np.random.seed(21)
        indices = np.random.randint(low=0, high=len(self.dataset), size=21)
        train_dataset = torch.utils.data.Subset(self.dataset, indices[6:])
        val_dataset = torch.utils.data.Subset(self.dataset, indices[:6])

        trainer = VisionClassifierTrainer(
            model_name=self.model_name,
            train=train_dataset,
            test=val_dataset,
            output_dir="./out/",
            lr=2e-5,
            model=ViTForImageClassification.from_pretrained(
                self.model_name,
                num_labels=3,
                ignore_mismatched_sizes=True
            ),
            feature_extractor=self.feature_extractor,
        )
        return trainer.train()
