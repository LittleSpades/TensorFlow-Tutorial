
from __future__ import absolute_import, division, print_function, unicode_literals


# TensorFlow e tf.keras
import tensorflow as tf
from tensorflow import keras

# Bibliotecas de ajuda
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

#9 types of clothing
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

################################ Train ################################
print("Quantity of train images, size (px) = ", train_images.shape)

print("Quantity of train images: ",len(train_labels))

print("Training Vector: ",train_labels, "\n")
#######################################################################

################################ Test #################################
print("Quantity of test images, size (px) = ", test_images.shape)

print("Quantity of test images: ",len(test_labels))
#######################################################################

######################### Pre-Processing Data #########################
plt.figure()
plt.imshow(train_images[0]) #Train Image 1
plt.colorbar()  
plt.grid(False)
plt.savefig('Pre-Processed Data Test.png')

#The values must be in a range of 0 to 1 before feeding to the neural network model
train_images = train_images / 255.0
test_images = test_images / 255.0
#######################################################################


####################### Training Data Example #########################
plt.figure(figsize=(10,10))

for i in range(25):

    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])

plt.savefig('Training Data Example.png')
#######################################################################

############################ Model SetUp ##############################
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer='adam', 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

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


############################# Training ################################

# This method is "fit" tothe training data
print('\n\nTraining...\n')
model.fit(train_images, train_labels, epochs=5)

print('\nComputing Loss/Accuracy Values...\n')
# Loss and Accuracy of the trained model on itself
train_loss, train_acc = model.evaluate(train_images, train_labels)
# Loss and Accuracy of the trained model on the test dataset
test_loss, test_acc = model.evaluate(test_images, test_labels)

#Print Accuracy Values
print('\nTraining Accuracy:', train_acc, '\nTest Accuracy:', test_acc, '\n\n')
# Test Accuracy < Training Accuracy <= OVERFITTING
#######################################################################

########################### Predictions ###############################

# Matrix in witch each column have the probabilities for each class of a test
# (#Lines X #Columns) = (#Classes X #Tests)
predictions = model.predict(test_images)


print('Class Probabilities for the first Prediction:')

from prettytable import PrettyTable
t = PrettyTable(['Class', 'Prob'])

for i in range(10):
    t.add_row([class_names[i], predictions[0][i]])
print(t)

print('\n => The Trained Model Predicts that the class of the first Test is'
    ,np.argmax(predictions[0]),'!')
#######################################################################