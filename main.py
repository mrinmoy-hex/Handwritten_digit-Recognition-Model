import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Function to train a neural network model on the MNIST dataset

# Load the MNIST dataset, which contains images of handwritten digits

def train_model():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the pixel values of the images to be between 0 and 1
    # This helps the neural network train faster and more accurately
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)
    # Initialize a Sequential model, which is a linear stack of layers
    model = tf.keras.models.Sequential()

    # Add a Flatten layer to reshape the 28x28 pixel images into a 1D array of 784 pixels
    # This is necessary because Dense layers expect 1D input
    model.add(tf.keras.layers.Flatten(input_shape=(28, 28)))

    # Add a Dense (fully connected) layer with 128 neurons and ReLU activation function
    # ReLU is commonly used in hidden layers to introduce non-linearity
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    # Add another Dense layer with 128 neurons and ReLU activation
    # This layer also helps the model learn complex patterns in the data
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    # Add the output layer with 10 neurons (one for each digit 0-9)
    # The softmax activation function is used here to output probabilities for each class
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # Compile the model by specifying the optimizer, loss function, and evaluation metric
    # The 'adam' optimizer adjusts the weights of the network efficiently
    # 'sparse_categorical_crossentropy' is used as the loss function for multi-class classification
    # 'accuracy' is the metric used to evaluate the performance of the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Train the model on the training data for 3 epochs
    # An epoch is one complete pass through the training data
    # if model's accuracy is low, consider increasing the number of epochs
    model.fit(x_train, y_train, epochs=3)

    # Save the trained model to a file with the .keras extension
    # This allows you to load and use the model later without retraining
    model.save('handwritten_model.keras')

# Load the saved model from the file
# This is useful for making predictions or further evaluation
model = tf.keras.models.load_model('handwritten_model.keras')

# loss, accuracy = model.evaluate(x_test, y_test)

# print(f"Loss: {loss}")
# print(f"Accuracy: {accuracy}")

image_number = 1
while os.path.isfile(f"data/digit{image_number}.png"):
    try:
        img = cv2.imread(f"data/digit{image_number}.png")[:,:,0]
        img = np.invert(np.array([img]))
        prediction = model.predict(img)
        print(f"This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img[0], cmap=plt.cm.binary)
        plt.show()
    except:
        print("May be there's some error :(")
    finally:
        image_number += 1