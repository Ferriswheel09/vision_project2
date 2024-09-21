import os
from PIL import Image
from transformers import AutoImageProcessor, ResNetForImageClassification
import torch
import torch.nn.functional as F

# Path to the directory containing images
directory = "./images/"

# Load the processor and model
processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

# Loop through all files in the directory
for filename in os.listdir(directory):
    image_path = os.path.join(directory, filename)

    # Load the image using Pillow
    image = Image.open(image_path)

    # Prepare the image for the model
    inputs = processor(image, return_tensors="pt")

    # Run the model and get the logits (raw scores)
    with torch.no_grad():
        logits = model(**inputs).logits

    # Apply softmax to get probabilities (confidence scores)
    probs = F.softmax(logits, dim=-1)

    # Get the predicted label and its confidence
    predicted_label = logits.argmax(-1).item()
    confidence = probs[0, predicted_label].item()

    # Print results for each image
    print(f"Image: {filename}")
    print(f"Predicted Label: {model.config.id2label[predicted_label]}")
    print(f"Confidence: {confidence:.4f}")
    print("-" * 40)
