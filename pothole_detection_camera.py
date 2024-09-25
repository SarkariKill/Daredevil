import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('pothole_detector_model.h5')

# Define the video capture object (0 is the default camera)
cap = cv2.VideoCapture(0)

# Preprocess function (adjust based on your model’s input size and preprocessing steps)
def preprocess_frame(frame):
    # Resize frame to the input size of the model (e.g., 224x224)
    frame_resized = cv2.resize(frame, (224, 224))
    
    # Normalize the frame if necessary (e.g., scale pixel values)
    frame_normalized = frame_resized / 255.0
    
    # Expand dimensions to match model input shape (1, 224, 224, 3)
    frame_expanded = np.expand_dims(frame_normalized, axis=0)
    
    return frame_expanded

# Loop to continuously get frames from the camera feed
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    # Preprocess the frame
    processed_frame = preprocess_frame(frame)

    # Make predictions with the model
    prediction = model.predict(processed_frame)
    
    # Example: assuming the model output is binary (1 for pothole, 0 for no pothole)
    if prediction > 0.5:
        label = 'Pothole Detected'
    else:
        label = 'No Pothole'

    # Display the prediction on the frame
    cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

    # Show the frame with prediction
    cv2.imshow('Pothole Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close OpenCV windows
cap.release()
cv2.destroyAllWindows()