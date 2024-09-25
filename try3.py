import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8s.pt')

# Function to calculate distance using the pinhole camera model
def calculate_distance(focal_length, real_object_height, object_height_in_image):
    return (real_object_height * focal_length) / object_height_in_image

# Real object heights in meters (example for known objects)
object_heights = {
    'person': 1.7,   # Average height of a person
    'bicycle': 1.0,  # Average height of a bicycle
    'car': 1.5,      # Average height of a car
}

# Focal length in pixels
focal_length = 1010

# Streamlit layout
st.title("Live YOLOv8 Object Detection")

# Checkbox to start/stop the camera
run = st.checkbox('Run Camera', value=False)

# Placeholder for displaying video
FRAME_WINDOW = st.empty()

if run:
    st.write("Camera is running...")
    cap = cv2.VideoCapture(0)  # Open the camera feed

    while run:
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to grab frame")
            break

        # Perform object detection
        results = model(frame)

        # Extract bounding boxes and labels
        for result in results:
            for box in result.boxes:  # Iterate through detected objects
                label = model.names[int(box.cls)]  # Convert class index to label name
                confidence = box.conf.item()
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0].tolist())

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

        # Convert the frame to RGB for Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Display the frame in Streamlit
        FRAME_WINDOW.image(frame_rgb, channels="RGB")

    cap.release()  # Release the camera when done
else:
    st.write("Camera is stopped.")

