from ultralytics import YOLO
import cv2
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')

# Open the webcam (0 is usually the default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video stream from the webcam.")
    exit()

# Function to provide voice feedback
def speak(text):
    engine.say(text)
    engine.runAndWait()

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

    # Store detected objects to avoid repeating the same message
    spoken_objects = set()

    # Print and provide voice feedback for detected objects
    print("Detected objects in the current frame:")
    for obj_id in detected_objects:
        object_name = object_names[int(obj_id)]
        print(object_name)  # Get the object name by ID

        # Speak the detected object if it has not been spoken yet
        if object_name not in spoken_objects:
            speak(f"I see a {object_name}")
            spoken_objects.add(object_name)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
