import os                        # Importing the 'os' module to access the operating system functionalities.
from symbol import tfpdef       # Importing the 'tfpdef' module from the 'symbol' package.
import cv2                       # Importing the 'cv2' module for image processing using OpenCV.
import numpy as np               # Importing the 'numpy' module to work with arrays and matrices.
import json                      # Importing the 'json' module for working with JSON data.
import tensorflow as tf          # Importing the TensorFlow library.
from tensorflow import keras     # Importing the 'keras' package from TensorFlow.
from keras.utils import to_categorical

print('1')                        

# Define the paths to the dataset directory and the annotation directory
dataset_dir = "C:/Users/juanc/fiftyone/coco-2017/validation/data"    # Setting the path of the dataset directory.
annotation_dir = "C:/Users/juanc/fiftyone/coco-2017/validation/labels.json"    # Setting the path of the annotation directory.
print('2')                        

# Define a function to load the annotations from a file
def load_annotations_file(annotation_file):    # Defining a function 'load_annotations_file' that takes the path of an annotation file as an argument.
    with open(annotation_file, "r") as f:    # Opening the annotation file in read-only mode.
        data = json.load(f)    # Loading the JSON data from the file.
    annotations = []    # Initializing an empty list to store the annotations.
    for image in data["images"]:    # Looping through each image in the JSON data.
        file_name = image["file_name"]    # Extracting the file name of the image.
        id = image["id"]    # Extracting the ID of the image.
        annotation = None    # Initializing a variable 'annotation' to None.
        for ann in data["annotations"]:    # Looping through each annotation in the JSON data.
            if ann["image_id"] == id:    # If the annotation belongs to the current image.
                annotation = ann["category_id"]    # Extracting the category ID of the annotation.
                break    # Stop searching for annotations for the current image.
        if annotation is not None:    # If there is at least one annotation for the current image.
            annotations.append((file_name, annotation))    # Append the file name and category ID to the 'annotations' list.
            print('2.1')
    return annotations    # Return the list of annotations.


print('3')                       


annotations = load_annotations_file(annotation_dir)
print('4')                        

# Create numpy arrays for the images and labels
images = np.zeros((len(annotations), 128, 64, 3), dtype=np.uint8)    # Creating an empty numpy array to store the images with the specified dimensions and data type.
labels = np.zeros((len(annotations), 1), dtype=np.uint8)    # Creating an empty numpy array to store the labels with the specified dimensions and

print('5')
# Loop through the annotations and load the corresponding images and labels
# Loop through the annotations and load and resize the images
for i, annotation in enumerate(annotations):
    # Create the path to the image by joining the dataset directory and the image name from the annotation
    image_path = os.path.join(dataset_dir, annotation[0])
    # Get the label for the image from the annotation
    label = annotation[1]

    # Load the image and resize it to the required size
    image = cv2.imread(image_path)
    if image is None:
        # If the image is None, print an error message and continue to the next image
        print(f"Could not read image {image_path}")
        continue
    # Resize the image to 64x128
    image = cv2.resize(image, (64, 128))

    # Store the image and label in the numpy arrays
    images[i] = image
    labels[i] = label
    print('6')

# Split the dataset into training and testing sets
num_train = int(0.8 * len(annotations))
num_test = len(annotations) - num_train

x_train = images[:num_train]
y_train = labels[:num_train]

x_test = images[num_train:]
y_test = labels[num_train:]


# Define the model architecture
model = keras.Sequential([
    # Add a 2D convolutional layer with 32 filters, a 3x3 kernel size, and ReLU activation
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 64, 3)),
    # Add a 2D max pooling layer with a 2x2 pool size
    keras.layers.MaxPooling2D((2, 2)),
    # Add another 2D convolutional layer with 64 filters, a 3x3 kernel size, and ReLU activation
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    # Add another 2D max pooling layer with a 2x2 pool size
    keras.layers.MaxPooling2D((2, 2)),
    # Add another 2D convolutional layer with 128 filters, a 3x3 kernel size, and ReLU activation
    keras.layers.Conv2D(128, (3, 3), activation='relu'),
    # Add another 2D max pooling layer with a 2x2 pool size
    keras.layers.MaxPooling2D((2, 2)),
    # Flatten the output from the previous layer
    keras.layers.Flatten(),
    # Add a fully connected layer with 512 units and ReLU activation
    keras.layers.Dense(512, activation='relu'),
    # Add a dropout layer with a rate of 0.5
    keras.layers.Dropout(0.5),
    # Add a final fully connected layer with 91 units and softmax activation
    keras.layers.Dense(91, activation='softmax')
])

    

# Compile the model with Adam optimizer, sparse categorical crossentropy loss, and accuracy metric
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

y_train = keras.utils.to_categorical(y_train, num_classes=91)
y_test = keras.utils.to_categorical(y_test, num_classes=91)

# Train the model with the training data for 20 epochs and validate with the testing data
model.fit(x_train, y_train, epochs=20, batch_size=32, validation_data=(x_test, y_test))

# Save the model to the specified path
model.save('C:/Users/juanc/Documents/seniordesigncode/modelresults')