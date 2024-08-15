# Handwritten Digit Recognition

A model that trains a model to recognize handwritten digits using the MNIST data set. Then it loads external files and uses the neural network to predict what digits they are.

## Handling Digit Recognition Failures
If the model fails to recognize digits accurately, consider the following solutions:

1. **Increase Training Epochs**: More training epochs can help the model learn better. Try increasing the number of epochs in the `model.fit()` method.
2. **Image Preprocessing**: Ensure that input images are preprocessed consistently with the training images. For example, ensure images are properly normalized and resized.
3 **Augment Data**: To improve model robustness, you can use data augmentation techniques
to generate variations of the training images, which can help the model generalize better.

4. **Tune Hyperparameters**: Adjust hyperparameters such as the number of neurons in the Dense layers, learning rate of the optimizer, or the choice of activation functions.
5. **Check Model Architecture**: Experiment with different architectures, such as adding more layers or using different types of layers (e.g., Convolutional layers).
6. **Ensure Proper Image Format**: Verify that the images you are using for prediction are correctly formatted and aligned with the model’s expected input dimensions and normalization.

## Libraries Used:

- [TensorFlow](https://www.tensorflow.org/) for providing the tools to build and train the model.
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/) for the handwritten digits dataset.

## Resolve all Dependencies:
```sh
pip install -r requirements.txt
```

Feel free to add 28x28 pixel images into the digits directory!
