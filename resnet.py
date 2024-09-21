from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
import torch.nn.functional as F
from datasets import load_dataset

image_path = "./images/twohundred_corruption.jpg"
image = Image.open(image_path)

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

inputs = processor(image, return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

probs = F.softmax(logits, dim=-1)


# model predicts one of the 1000 ImageNet classes
predicted_label = logits.argmax(-1).item()
confidence = probs[0, predicted_label].item()

print(model.config.id2label[predicted_label])
print(f"Confidence: {confidence:.4f}")