from transformers import DetrFeatureExtractor, DetrForObjectDetection
from model.dataset_creation import CocoTeggsme
from pathlib import Path
from model.computer_vision_model import EggsViTModel
from PIL import Image


my_img_folder = Path("C:/Users/Genesis/Teggs-me/dataset/images")
my_ann_folder = Path("C:/Users/Genesis/Teggs-me/dataset/masks")
my_ann_file = Path("C:/Users/Genesis/Teggs-me/dataset/coco_instances.json")


class PretrainedModelPrediction:
    """This is a computer vision model in ImageClassification task wrapped in huggingface."""

    def __init__(self,
                 model_name_path):
        """This is a prediction manager. Here are two possibilities:
        1. Make a prediction with a huggingaface pretrained model,
        2. Make a prediction with a custom model
        """
        self.model_name = model_name_path
        self.img_class_pretrained = 'facebook/detr-resnet-101-dc5'
        self.model_output_path = "./my_model"
        self.feature_extractor_pretrained = DetrFeatureExtractor.from_pretrained(self.img_class_pretrained)
        self.model_pretrained = DetrForObjectDetection.from_pretrained(self.img_class_pretrained)

    def image_classification_classifier_pretrained(self, uploaded_file):
        """
        This method makes an image classification on a pretrained model.
        """
        img = Image.open(uploaded_file).convert("RGB")
        pixels = img.load()
        inputs = self.feature_extractor_pretrained(images=img, return_tensors="pt")
        outputs = self.model_pretrained(**inputs)

        # model predicts bounding boxes and corresponding COCO classes
        logits = outputs.logits
        bboxes = outputs.pred_boxes
        return logits, bboxes

    def image_classification_classifier_my_model(self, uploaded_file):
        """
        This method handles:
            - dataset for model training creation
            - model for training creation
            - model training
            - image classification.
        """
        my_feature_extractor = DetrFeatureExtractor.from_pretrained(self.model_output_path)
        my_model = DetrForObjectDetection.from_pretrained(self.model_output_path)
        # dataset definition
        dataset = CocoTeggsme(my_img_folder,
                              my_ann_file)
        # model definition
        my_parameters = {"model_name": self.img_class_pretrained,
                         "model_training_parameters": {"max_epochs": 1}, }
        model = EggsViTModel(my_parameters, dataset)
        model.model_training()
        model.save(self.model_output_path)
        img = Image.open(uploaded_file).convert("RGB")
        pixels = img.load()
        inputs = my_feature_extractor(images=img, return_tensors="pt")
        outputs = my_model(**inputs)

        # model predicts bounding boxes and corresponding COCO classes
        logits = outputs.logits
        bboxes = outputs.pred_boxes
        return logits, bboxes
