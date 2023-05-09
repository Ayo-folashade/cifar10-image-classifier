# Image Classification Model
This model is designed to classify images into 10 different classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck. It is built using the Keras library with TensorFlow.

# Dataset
The model is trained on the CIFAR-10 dataset, which consists of 50,000 training images and 10,000 test images. Each image is a 32x32 color image, belonging to one of the 10 classes mentioned above.

# Model Architecture
The model architecture consists of a convolutional neural network (CNN) with the following layers:

- Conv2D layer with 32 filters and a kernel size of (3,3)
- ReLU activation layer
- Conv2D layer with 32 filters and a kernel size of (3,3)
- ReLU activation layer
- MaxPooling2D layer with a pool size of (2,2)
- Dropout layer with a rate of 0.25
- Conv2D layer with 64 filters and a kernel size of (3,3)
- ReLU activation layer
- Conv2D layer with 64 filters and a kernel size of (3,3)
- ReLU activation layer
- MaxPooling2D layer with a pool size of (2,2)
- Dropout layer with a rate of 0.25
- Flatten layer
- Dense layer with 512 units and a ReLU activation
- Dropout layer with a rate of 0.5
- Dense layer with 10 units and a Softmax activation

# Training
The model was trained for 20 epochs with a batch size of 32 and an Adam optimizer. The categorical cross-entropy loss function was used as the loss function.

# Evaluation
The model achieved an accuracy of approximately 75% on the test set.

# Usage
To use the CNN-SVM classifier, you can run the [cnn_svm_classifier.py](cnn_svm_classifier.py) script included in this repository. This script will train the model on the CIFAR-10 training dataset, and then evaluate the model on the CIFAR-10 test dataset.
```
python cnn_svm_classifier.py
```

The script will output the accuracy of the trained model on the test dataset.

You can also use the [cnn_svm_classifier.py](cnn_svm_classifier.py) notebook included in the notebooks directory to explore the code and visualize the results.

To test the model on new images, you can add your images to the [new_test_images](new_test_images) directory and modify the [cnn_svm_classifier.py](cnn_svm_classifier.py) script to load your images. Then, run the script to classify your images.

# Requirements
To run the code in this repository, you will need to install the following Python packages:

- tensorflow
- keras 
- scikit-learn 
- matplotlib 
- numpy

You can install these packages using the [requirements.txt](requirements.txt) file included in this repository. To do so, simply run the following command:
```
pip install -r requirements.txt
```

## License

This code is released under the MIT License. See [LICENSE](LICENSE) for more information.
