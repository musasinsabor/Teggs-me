from transformers import ViTFeatureExtractor, ViTForImageClassification
from typing import Dict
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from pathlib import Path
from dataset_creation import CocoTeggsme
import numpy as np
import torch


class EggsViTModel:
    """This is a computer vision model in ImageClassification task wrapped in huggingface."""
    def __init__(self,
                 parameters):
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
        self.dataset = self.dataset_obj()
        self.model = self.pretrained_model()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name)

    def dataset_obj(self):
        """
        This method uses the dataset info to create a dataset instance.
        :returns a Dataset object
        """
        my_img_folder = Path("C:/Users/Genesis/Teggs-me/dataset/images")
        my_ann_folder = Path("C:/Users/Genesis/Teggs-me/dataset/images/masks")
        my_ann_file = Path("C:/Users/Genesis/Teggs-me/dataset/coco_instances.json")
        dataset = CocoTeggsme(my_img_folder, my_ann_folder, my_ann_file, self.feature_extractor)
        return dataset

    def pretrained_model(self):
        """
        This method specify a pretrained ViModel.
        """
        pt_model = ViTForImageClassification.from_pretrained(self.model_name, num_labels=len(self.num_labels))
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
            fp16=True,
            model=ViTForImageClassification.from_pretrained(
                self.model,
                num_labels=3,
            ),
            feature_extractor=self.feature_extractor,
        )
        return trainer.train()
