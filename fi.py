from ultralytics import YOLO

# Load the YOLOv8 model
model = YOLO('yolov8s.pt')

# Print the list of class names (object categories)
print(model.names)
