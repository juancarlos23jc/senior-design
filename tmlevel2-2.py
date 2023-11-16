import cv2
import numpy as np
import tensorflow as tf
import smtplib
#import time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage

#last_email_time = 0
#email_delay = 60

#sending email
fromEmail = 'tamuteam16@gmail.com'
#password        
fromEmailPassword = 'dbnz cafr cttp kgxu'#ECEN404!

# recieving email
toEmail = 'juancarloscruz2324@gmail.com'

def sendEmail(image):
    # Convert the image from BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Encode the image as a JPEG
    ret, jpeg_image = cv2.imencode('.jpg', image)
    if not ret:
        raise Exception('Image encoding failed')
    
    # Convert the encoded image to bytes
    image_bytes = jpeg_image.tobytes()

    # Set up the email
    msgRoot = MIMEMultipart('related')
    msgRoot['Subject'] = 'Security Update'
    msgRoot['From'] = fromEmail
    msgRoot['To'] = toEmail
    msgRoot.preamble = 'Raspberry pi security camera update'

    msgAlternative = MIMEMultipart('alternative')
    msgRoot.attach(msgAlternative)
    msgText = MIMEText('Smart security cam found object')
    msgAlternative.attach(msgText)

    msgText = MIMEText('<img src="cid:image1">', 'html')
    msgAlternative.attach(msgText)

    msgImage = MIMEImage(image_bytes, 'jpeg')  # Specify the correct image format
    msgImage.add_header('Content-ID', '<image1>')
    msgRoot.attach(msgImage)

    # Send the email
    smtp = smtplib.SMTP('smtp.gmail.com', 587)
    smtp.starttls()
    smtp.login(fromEmail, fromEmailPassword)
    smtp.sendmail(fromEmail, toEmail, msgRoot.as_string())
    smtp.quit()      
# Load the saved model
model = tf.keras.models.load_model('C:/Users/juanc/Documents/seniordesigncode/model4')

# Define the labels
labels = ['not a person', 'person']

# Set the prediction threshold
threshold = 0.75



# Open the video capture device
#cap = cv2.VideoCapture(0)
url = "http://172.20.10.11:5000/video_feed"
cap = cv2.VideoCapture(url)


# Initialize flag to check if an email has been sent
#email_sent = False

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
        sendEmail(image)
        #last_email_time = time.time()
        
        
        
            
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

