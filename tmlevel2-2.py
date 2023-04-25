import cv2
import numpy as np
import tensorflow as tf

# Load the saved model
model = tf.keras.models.load_model('C:/Users/juanc/Documents/seniordesigncode/model4')

# Define the labels
labels = ['not a person', 'person']

# Set the prediction threshold
threshold = 0.99

# Open the video capture device
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    
    # Preprocess the image for the model
    image = cv2.resize(frame, (64, 128))
    image = np.expand_dims(image, axis=0)
    
    # Predict the class of the image
    prediction = model.predict(image)
    prediction_class = np.argmax(prediction)
    prediction_probability = np.max(prediction)
    
    # Check if a person is detected in the current frame
    if prediction_class == 1 and prediction_probability > threshold:
        prediction_label = labels[prediction_class]
    else:
        prediction_label = labels[0]
    
    # Draw the label and probability on the frame
    label_text = f"{prediction_label} ({prediction_probability:.2f})"
    cv2.putText(frame, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Video', frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture device and close the window
cap.release()
cv2.destroyAllWindows()
