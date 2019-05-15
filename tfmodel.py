from __future__ import absolute_import, division, print_function, unicode_literals



import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tutorialfunctions

# Libraries
import numpy as np
from prettytable import PrettyTable

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras



def modelSetup(_train_images, _train_labels):
    ############################ Model SetUp ##############################

    _model = keras.Sequential([
        keras.layers.Flatten(input_shape=(28, 28)),
        keras.layers.Dense(128, activation=tf.nn.relu),
        keras.layers.Dense(10, activation=tf.nn.softmax)
    ])

    _model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    exit = raw_input("\nModel Created...\nPress any key to Exit! ")

    return _model

    # OPTIMIZER: How the model is updated based on the data it sees and its 
    #            loss function.
    #
    # LOSS FUNCTION: Measures how accurate the model is during training.
    #                We want to minimize this function to "steer" the model 
    #                in the right direction.
    #
    # METRICS: Monitors the training and testing steps. 
    #          -> 'accuracy': The fraction of the images that are correctly 
    #                         classified.
    #######################################################################

def modelTraining(_model, _train_images, _train_labels):


    ############################# Training ################################
    # This method is "fit" to the training data
    print('Training...\n')
    _model.fit(_train_images, _train_labels, epochs=5)

    exit = raw_input("\nPress any key to Exit! ")

    return _model

    

def aculoss(_model, _train_images, _train_labels, _test_images, _test_labels):
    
    print('\nComputing Loss/Accuracy Values...\n')
    # Loss and Accuracy of the trained model on itself
    train_loss, train_acc = _model.evaluate(_train_images, _train_labels)
    # Loss and Accuracy of the trained model on the test dataset
    test_loss, test_acc = _model.evaluate(_test_images, _test_labels)

    #Print Accuracy Values
    print('\nTraining Accuracy:', train_acc, '\nTest Accuracy:', test_acc, '\n\n')
    # Test Accuracy < Training Accuracy <= OVERFITTING

    exit = raw_input("\nPress any key to Exit! ")
 
def predictions(_model, _test_images, _class_names, _test_labels):
    
    ########################### Predictions ###############################

    # Matrix in witch each column have the probabilities for each class of a test
    # (#Lines X #Columns) = (#Classes X #Tests)
    predictions = _model.predict(_test_images)


    print('Class Probabilities for the first Prediction:')

    t = PrettyTable(['Class', 'Prob'])

    for i in range(10):
        t.add_row([_class_names[i], predictions[0][i]])
    print(t)

    print('\n => The Trained Model Predicts that the class of the first Test is:'
        , _class_names[np.argmax(predictions[0])])

    print('\n => The label of the Test is:', _class_names[_test_labels[0]])
    #######################################################################


    # Plot the first X test images, their predicted label, and the true label
    # Color correct predictions in blue, incorrect predictions in red
    num_rows = 5
    num_cols = 3
    num_images = num_rows*num_cols
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))
    for i in range(num_images):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        tutorialfunctions.plot_image(i, predictions, _test_labels, _test_images, _class_names)
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        tutorialfunctions.plot_value_array(i, predictions,  _test_labels)

    plt.savefig('SetOfPredictions.png')
    print("Created 'SetOfPredictions.png'")
    
    exit = raw_input("\nPress any key to Exit! ")

    