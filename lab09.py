import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T
import cv2
import numpy as np
from google.colab import files

'''from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights

# Get the COCO class names
coco_classes = FasterRCNN_ResNet50_FPN_Weights.DEFAULT.meta['categories']
print(coco_classes)'''
COCO_CLASSES = [
    "__background__", "person", "bicycle", "car", "motorcycle", "airplane", "bus",
    "train", "truck", "boat", "traffic light", "fire hydrant", "N/A", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant",
    "bear", "zebra", "giraffe", "N/A", "backpack", "umbrella", "N/A", "N/A", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "N/A", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "N/A",
    "dining table", "N/A", "N/A", "toilet", "N/A", "TV", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "N/A", "book", "clock",
    "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
]

model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval() 
print("Please upload an image for detection:")
uploaded = files.upload()
image_path = list(uploaded.keys())[0] 

image = Image.open(image_path).convert("RGB")

transform = T.Compose([
    T.ToTensor() 
])
image_tensor = transform(image)

with torch.no_grad():  
    predictions = model([image_tensor]) 

# Extract predictions
boxes = predictions[0]['boxes'] 
labels = predictions[0]['labels']
scores = predictions[0]['scores']  

image_np = np.array(image)
confidence_threshold = 0.5

for i, box in enumerate(boxes):
    if scores[i] > confidence_threshold: 
        x1, y1, x2, y2 = map(int, box) 
        label_index = labels[i].item()  
        label_name = COCO_CLASSES[label_index]
        score = scores[i].item() 

        cv2.rectangle(image_np, (x1, y1), (x2, y2), (0, 255, 0), 2)
        text = f"{label_name}: {score:.2f}"
        cv2.putText(image_np, text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

plt.figure(figsize=(12, 8))
plt.imshow(image_np)
plt.axis("off")
plt.title("Object Detection Results")
plt.show()
