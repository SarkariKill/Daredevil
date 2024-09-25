import cv2
import torch

# Load the pre-trained YOLOv5 model (using PyTorch Hub)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

# Function to calculate distance using the pinhole camera model
def calculate_distance(focal_length, real_object_height, object_height_in_image):
    return (real_object_height * focal_length) / object_height_in_image

# Real object heights in meters (example for known objects)
object_heights = {
    'person': 1.7,  # average height of a person
    'car': 1.5,     # average height of a car
    'bottle': 0.25  # average height of a bottle
}

# Focal length in pixels (this can be obtained by calibrating your camera)
focal_length = 19729 

# Open the camera feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture image")
        break
    
    # Perform object detection
    results = model(frame)
    
    # Extract bounding boxes and labels
    detected_objects = results.pandas().xyxy[0]  # xyxy[0] returns a pandas dataframe
    
    for index, row in detected_objects.iterrows():
        label = row['name']
        confidence = row['confidence']
        x_min, y_min, x_max, y_max = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        
        # Calculate height of the detected object in the image
        object_height_in_image = y_max - y_min
        
        # Check if the object label has a known real-world height
        if label in object_heights:
            real_object_height = object_heights[label]
            
            # Calculate distance to the object
            distance = calculate_distance(focal_length, real_object_height, object_height_in_image)
            
            # Display the bounding box, label, confidence, and estimated distance
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            text = f'{label} {confidence:.2f} Dist: {distance:.2f}m'
            cv2.putText(frame, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Display the frame with annotations
    cv2.imshow('YOLO Object Detection', frame)
    
    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
