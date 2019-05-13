
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

print("Vector de treino: ",train_labels, "\n")
#######################################################################

################################ Test #################################
print("Quantity of test images, size (px) = ", test_images.shape)

print("Quantity of test images: ",len(test_labels))
#######################################################################

######################### Pre-Processing Data #########################
fig = plt.figure()
plt.imshow(train_images[0]) #Train Image 1
plt.colorbar()
plt.grid(False)
plt.show(fig)
#fig.savefig('test.png')
#######################################################################
