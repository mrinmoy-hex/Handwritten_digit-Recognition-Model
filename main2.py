import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Function to train the neural network model
def train_model():
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Normalize the pixel values
    x_train = tf.keras.utils.normalize(x_train, axis=1)
    x_test = tf.keras.utils.normalize(x_test, axis=1)

    # Build the model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),  # Flatten input
        tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer 1
        tf.keras.layers.Dense(128, activation='relu'),  # Hidden layer 2
        tf.keras.layers.Dense(10, activation='softmax')  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', 
                  loss='sparse_categorical_crossentropy', 
                  metrics=['accuracy'])

    # Train the model
    # An epoch is one complete pass through the training data
    # if model's accuracy is low, consider increasing the number of epochs
    model.fit(x_train, y_train, epochs=10)

    # Save the trained model
    model.save('handwritten_model.keras')

    # Evaluate the model
    loss, accuracy = model.evaluate(x_test, y_test)
    print(f"Loss: {loss}")
    print(f"Accuracy: {accuracy}")

    return model

# Function to load a pre-trained model
def load_model(model_path='handwritten_model.keras'):
    return tf.keras.models.load_model(model_path)

# Function to predict digits from images
def predict_digits(model):
    image_number = 1
    while os.path.isfile(f"data/digit{image_number}.png"):
        try:
            img = cv2.imread(f"data/digit{image_number}.png")[:,:,0]
            img = np.invert(np.array([img]))
            prediction = model.predict(img)
            print(f"This digit is probably a {np.argmax(prediction)}")
            plt.imshow(img[0], cmap=plt.cm.binary)
            plt.show()
        except Exception as e:
            print(f"Error processing image {image_number}: {e}")
        finally:
            image_number += 1

# Main code execution
if __name__ == "__main__":
    # Train and save the model (uncomment if you want to train again)
    # model = train_model()

    # Load the pre-trained model
    model = load_model()

    # Predict digits using the loaded model
    predict_digits(model)
