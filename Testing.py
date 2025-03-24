import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import supervision as sv
from ultralytics import YOLOv10
import torch


# Load the model
model = YOLOv10('new10n.pt')

bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

cap = cv2.VideoCapture("http://192.168.215.64:8080/video")

if not cap.isOpened():
    print("Unable to read camera feed")

img_counter = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    # Flip the frame horizontally to create a mirror effect
    frame = cv2.flip(frame, 1)

    # Run detection
    results = model(frame)[0]
    
    # Get detections with confidence scores
    detections = sv.Detections.from_ultralytics(results)
    
    # Create labels with confidence scores
    labels = [
        f"{results.names[class_id]} {confidence:.2f}"
        for class_id, confidence
        in zip(detections.class_id, detections.confidence)
    ]
    
    # Annotate the frame
    annotated_image = bounding_box_annotator.annotate(
        scene=frame, 
        detections=detections
    )
    
    # Add labels with confidence scores
    annotated_image = label_annotator.annotate(
        scene=annotated_image,
        detections=detections,
        labels=labels
    )
    
    # Display the result
    cv2.imshow('Webcam', annotated_image)

    k =cv2.waitKey(1)

    if k%256 == 27:
        print("Escape hit, closing...")
        break

cap.release()
cv2.destroyAllWindows()