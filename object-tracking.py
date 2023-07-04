import cv2
import torch
from torchvision import models, transforms

model = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

transform = transforms.Compose([
    transforms.ToTensor()
])

classes = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
    'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
    'N/A', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
    'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'N/A',
    'backpack', 'umbrella', 'N/A', 'N/A', 'handbag', 'tie', 'suitcase',
    'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
    'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle',
    'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog',
    'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
    'N/A', 'dining table', 'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop',
    'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
    'sink', 'refrigerator', 'N/A', 'book', 'clock', 'vase', 'scissors',
    'teddy bear', 'hair drier', 'toothbrush'
]

image = cv2.imread('imagem.jpeg')

input_tensor = transform(image)
input_batch = input_tensor.unsqueeze(0)

with torch.no_grad():
    prediction = model(input_batch)

filtered_boxes = []
for i in range(prediction[0]['boxes'].shape[0]):
    if prediction[0]['scores'][i] > 0.8:
        filtered_boxes.append(prediction[0]['boxes'][i])
        class_index = int(prediction[0]['labels'][i])
        class_label = classes[class_index]
        score = float(prediction[0]['scores'][i])

        x, y, w, h = filtered_boxes[-1]

        cv2.rectangle(image, (int(x), int(y)), (int(w), int(h)), (0, 255, 0), 2)

        label = f"{class_label}: {score:.2f}"
        cv2.putText(image, label, (int(x), int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

cv2.imshow('Object Tracking', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
