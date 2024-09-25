import streamlit as st
import cv2
from streamlit_lottie import st_lottie 
import torch
from ultralytics import YOLO


# Load the pre-trained YOLOv8 model
model = YOLO('yolov8s.pt')  # You can replace 'yolov8s.pt' with 'yolov8m.pt' or 'yolov8l.pt' depending on your preference

# Function to calculate distance using the pinhole camera model
def calculate_distance(focal_length, real_object_height, object_height_in_image):
    return (real_object_height * focal_length) / object_height_in_image


# Real object heights in meters (example for known objects)
object_heights = {
   'person': 1.7,             # average height of a person
    'bicycle': 1.0,            # average height of a bicycle
    'car': 1.5,                # average height of a car
    'motorcycle': 1.1,         # average height of a motorcycle
    'airplane': 3.0,           # average height of an airplane (small)
    'bus': 3.0,                # average height of a bus
    'train': 3.5,              # average height of a train
    'truck': 3.5,              # average height of a truck
    'boat': 2.0,               # average height of a boat
    'traffic light': 3.0,      # average height of a traffic light
    'fire hydrant': 0.75,      # average height of a fire hydrant
    'stop sign': 2.1,          # average height of a stop sign
    'parking meter': 1.5,      # average height of a parking meter
    'bench': 0.9,              # average height of a bench
    'bird': 0.3,               # average height of a bird (small)
    'cat': 0.3,                # average height of a cat
    'dog': 0.6,                # average height of a dog
    'horse': 1.6,              # average height of a horse
    'sheep': 1.0,              # average height of a sheep
    'cow': 1.4,                # average height of a cow
    'elephant': 3.3,           # average height of an elephant
    'bear': 2.5,               # average height of a bear
    'zebra': 1.5,              # average height of a zebra
    'giraffe': 5.5,            # average height of a giraffe
    'backpack': 0.5,           # average height of a backpack
    'umbrella': 1.0,           # average height of an umbrella (folded)
    'handbag': 0.3,            # average height of a handbag
    'tie': 0.5,                # average length of a tie
    'suitcase': 0.7,           # average height of a suitcase
    'frisbee': 0.03,           # average height of a frisbee (thickness)
    'skis': 1.7,               # average length of skis
    'snowboard': 1.5,          # average length of a snowboard
    'sports ball': 0.25,       # average height of a sports ball (football/basketball)
    'kite': 0.8,               # average height of a kite
    'baseball bat': 1.1,       # average length of a baseball bat
    'baseball glove': 0.3,     # average height of a baseball glove
    'skateboard': 0.1,         # average height of a skateboard (thickness)
    'surfboard': 2.0,          # average length of a surfboard
    'tennis racket': 0.7,      # average height of a tennis racket
    'bottle': 0.25,            # average height of a bottle
    'wine glass': 0.2,         # average height of a wine glass
    'cup': 0.1,                # average height of a cup
    'fork': 0.02,              # average height of a fork (thickness)
    'knife': 0.03,             # average height of a knife (thickness)
    'spoon': 0.02,             # average height of a spoon (thickness)
    'bowl': 0.08,              # average height of a bowl
    'banana': 0.2,             # average length of a banana
    'apple': 0.1,              # average height of an apple
    'sandwich': 0.05,          # average height of a sandwich
    'orange': 0.08,            # average height of an orange
    'broccoli': 0.2,           # average height of broccoli
    'carrot': 0.2,             # average height of a carrot
    'hot dog': 0.15,           # average height of a hot dog
    'pizza': 0.03,             # average height of a pizza (thickness)
    'donut': 0.05,             # average height of a donut
    'cake': 0.15,              # average height of a cake
    'chair': 1.0,              # average height of a chair
    'couch': 1.0,              # average height of a couch
    'potted plant': 0.5,       # average height of a potted plant
    'bed': 0.6,                # average height of a bed
    'dining table': 0.75,      # average height of a dining table
    'toilet': 0.8,             # average height of a toilet
    'tv': 0.6,                 # average height of a television
    'laptop': 0.3946,            # average height of a laptop (thickness)
    'mouse': 0.05,             # average height of a computer mouse
    'remote': 0.02,            # average height of a remote control (thickness)
    'keyboard': 0.03,          # average height of a keyboard (thickness)
    'cell phone': 0.02,        # average height of a cell phone (thickness)
    'microwave': 0.4,          # average height of a microwave
    'oven': 0.8,               # average height of an oven
    'toaster': 0.3,            # average height of a toaster
    'sink': 0.4,               # average height of a sink
    'refrigerator': 1.8,       # average height of a refrigerator
    'book': 0.03,              # average height of a book (thickness)
    'clock': 0.2,              # average height of a clock
    'vase': 0.3,               # average height of a vase
    'scissors': 0.2,           # average height of scissors
    'teddy bear': 0.4,         # average height of a teddy bear
    'hair drier': 0.3,         # average height of a hair dryer
    'toothbrush': 0.2          # average height of a toothbrush
}


# Calibrate camera to get the focal length
focal_length = 1010


# Navigation Bar
# Define the navigation options
nav_options = {
    "Home": "Home",
    "Object Classification": "Object Classification",
    "Distance Estimation": "Distance Estimation", 
    "Pothole Detection": "Pothole Detection"
}

#For Lootie Aimation
def load_lottiefile(filepath: str):
    with open(filepath,"r") as f:
        return json.load(f)
    
# Create a sidebar with a selectbox for navigation
selected_page = st.sidebar.selectbox("Navigation", list(nav_options.keys()))

if selected_page == "Home":
    st.title("Hello There")
    
elif selected_page == "Object Classification":
    # Streamlit layout
    st.title("Live YOLOv8 Object Detection")

    # Checkbox to start/stop the camera
    run = st.checkbox('Run Camera', value=False)

    # Placeholder for displaying video
    FRAME_WINDOW = st.empty()

    # Initialize the camera variable
    cap = cv2.VideoCapture(0)  # Open the camera feed

    if run:
        st.write("Camera is running...")
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

        # When the loop ends, release the camera
        cap.release()  # Release the camera when done

    else:
        # If the camera is not running, release the camera
        cap.release()
        st.write("Camera is stopped.")