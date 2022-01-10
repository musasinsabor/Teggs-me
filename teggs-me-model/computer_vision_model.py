from transformers import ViTFeatureExtractor, ViTForImageClassification
from typing import Dict
from hugsvision.nnet.VisionClassifierTrainer import VisionClassifierTrainer


class EggsViTModel:
    """This is a computer vision model in ImageClassification task wrapped in huggingface."""
    def __init__(self,
                 dataset,
                 parameters):
        self.dataset = dataset
        self.parameters: Dict = parameters
        self.model_name = parameters["model_name"]
        self.model = self.pretrained_model_wrapped()
        self.labels = parameters["labels"]

    def pretrained_model_wrapped(self):
        pt_model = ViTForImageClassification.from_pretrained(self.model_name, num_labels=len(self.labels))
        return pt_model

    def model_training(self):
        train = self.dataset["train"]
        test = self.dataset["test"]
        label2id = self.dataset["label2id"]
        id2label = self.dataset["id2label"]

        trainer = VisionClassifierTrainer(
            model_name="MyKvasirV2Model",
            train=train,
            test=test,
            output_dir="./out/",
            lr=2e-5,
            fp16=True,
            model=ViTForImageClassification.from_pretrained(
                self.model,
                num_labels=len(label2id),
                label2id=label2id,
                id2label=id2label
            ),
            feature_extractor=ViTFeatureExtractor.from_pretrained(
                self.model,
            ),
        )
        return trainer.train()
