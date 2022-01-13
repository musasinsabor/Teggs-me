from transformers import ViTFeatureExtractor, ViTForImageClassification
from typing import Dict
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer
from pathlib import Path
from dataset_creation import CocoTeggsme


class EggsViTModel:
    """This is a computer vision model in ImageClassification task wrapped in huggingface."""
    def __init__(self,
                 parameters):
        self.parameters: Dict = parameters
        self.model_name = parameters["model_name"]
        self.labels = parameters["labels"]
        self.dataset = self.dataset_obj()
        self.model = self.pretrained_model()
        self.feature_extractor = ViTFeatureExtractor.from_pretrained(self.model_name)

    def dataset_obj(self):
        """
        This method uses the dataset info to create a dataset instance.
        :returns a Dataset object
        """
        my_img_folder = Path("/teggs-me-dataset/images")
        my_ann_folder = Path("/teggs-me-dataset/images")
        my_ann_file = Path("/teggs-me-dataset/coco_instances.json")
        dataset = CocoTeggsme(my_img_folder, my_ann_folder, my_ann_file, self.feature_extractor)
        return dataset

    def pretrained_model(self):
        """
        This method specify a pretrained ViModel.
        """
        pt_model = ViTForImageClassification.from_pretrained(self.model_name, num_labels=len(self.labels))
        return pt_model

    def model_training(self):
        train = self.dataset["train"]
        test = self.dataset["test"]

        trainer = VisionClassifierTrainer(
            model_name=self.model_name,
            train=train,
            test=test,
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
