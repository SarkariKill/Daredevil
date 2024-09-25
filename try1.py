from ultralytics import YOLO
import cv2

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from the webcam.")
    exit()

# Loop to continuously get frames from the webcam
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame.")
        break

    # Perform prediction on the frame
    results = model.predict(source=frame, show=True)  # show=True to visualize predictions

    # Extract and print detected objects
    detected_objects = results[0].boxes.cls  # Detected object class IDs
    object_names = results[0].names  # Object names based on IDs

    # Print the detected objects with their class labels
    print("Detected objects in the current frame:")
    for obj_id in detected_objects:
        print(object_names[int(obj_id)])  # Get the object name by ID

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
