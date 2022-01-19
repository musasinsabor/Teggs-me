from transformers import DetrFeatureExtractor, DetrForObjectDetection
from model.dataset_creation import CocoTeggsme
from model.computer_vision_model import EggsViTModel
from hugsvision.inference.VisionClassifierInference import VisionClassifierInference
from PIL import Image
import torch
import matplotlib.pyplot as plt
import os

directory = os.path.dirname(os.path.realpath(__file__))
my_data = "dataset"
my_imgs = "images"
my_masks = "masks"
my_file = "coco_instances.json"
my_img_folder = os.path.join(directory, my_data, my_imgs)
my_ann_folder = os.path.join(directory, my_data, my_masks)
my_ann_file = os.path.join(directory, my_data, my_file)


class PretrainedObjectDetectionModelPrediction:
    """This is a computer vision model in ImageClassification task wrapped in huggingface."""

    def __init__(self):
        """This is a prediction manager. Here are two possibilities:
        1. Make a prediction with a huggingaface pretrained model,
        2. Make a prediction with a custom model
        """
        self.img_class_pretrained = 'facebook/detr-resnet-101-dc5'
        self.model_output_path = "./my_model"
        self.feature_extractor_pretrained = DetrFeatureExtractor.from_pretrained(self.img_class_pretrained)
        self.model_pretrained = DetrForObjectDetection.from_pretrained(self.img_class_pretrained)
        self.COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
                       [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]
        self.id2label = self.model_pretrained.config.id2label

    def image_classification_classifier_pretrained(self, uploaded_file):
        """
        This method makes an image classification on a pretrained model.
        """
        img = Image.open(uploaded_file).convert("RGB")
        pixels = img.load()
        inputs = self.feature_extractor_pretrained(images=img, return_tensors="pt")
        outputs = self.model_pretrained(**inputs)
        logits = outputs.logits
        bboxes = outputs.pred_boxes

        return logits, bboxes

    # for output bounding box post-processing
    def box_cxcywh_to_xyxy(self, x):
        x_c, y_c, w, h = x.unbind(1)
        b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
             (x_c + 0.5 * w), (y_c + 0.5 * h)]
        return torch.stack(b, dim=1)

    def rescale_bboxes(self, out_bbox, size):
        img_w, img_h = size
        b = self.box_cxcywh_to_xyxy(out_bbox)
        b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
        return b

    def plot_results(self, pil_img, prob, boxes):
        plt.figure(figsize=(16, 10))
        plt.imshow(pil_img)
        ax = plt.gca()
        colors = self.COLORS * 100
        for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
            ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin,
                                       fill=False, color=c, linewidth=3))
            cl = p.argmax()
            text = f'{self.id2label[cl.item()]}: {p[cl]:0.2f}'
            ax.text(xmin, ymin, text, fontsize=15,
                    bbox=dict(facecolor='yellow', alpha=0.5))
        plt.axis('off')
        plt.show()

    def visualize_predictions(self, image, outputs, threshold=0.9):
        # keep only predictions with confidence >= threshold
        probas = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probas.max(-1).values > threshold

        # convert predicted boxes from [0; 1] to image scales
        bboxes_scaled = self.rescale_bboxes(outputs.pred_boxes[0, keep].cpu(), image.size)

        # plot results
        results = self.plot_results(image, probas[keep], bboxes_scaled)
        return results

    def object_detection_classifier(self):
        """
        This method makes an image classification on a pretrained model.
        """
        classifier = VisionClassifierInference(
            feature_extractor=self.feature_extractor_pretrained,
            model=self.model_pretrained,
        )
        return classifier

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
