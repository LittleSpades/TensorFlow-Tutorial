from __future__ import absolute_import, division, print_function, unicode_literals



import ExploreTheData as etd
import DataExamples as dtex
import tfmodel as md

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Libraries
import numpy as np
from prettytable import PrettyTable
import os
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


################################ Init #################################

print(tf.__version__)

ex = 0

fashion_mnist = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 10 types of clothing
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
#######################################################################

while ex==0:

  import subprocess as sp
  tmp = sp.call('clear',shell=True)

  
  
  print("TensorFlow Tutorial\n\n")
  print("1 - Explore the Data")
  print("2 - PreProcess Data and Training Data Example")
  print("3 - Model Setup")
  print("4 - Model Training")
  print("5 - Accuracy")
  print("6 - Predictions")
  print("7 - Exit")

  while True:
    try:
      response = int(raw_input("What do you want to do (1-6)? "))
      break
    except ValueError:
      print('Oops!  That was no valid number.  Try again...')

  if response == 1 :
    tmp = sp.call('clear',shell=True)
    etd.explore(train_images, train_labels, test_images, test_labels)  
  elif response == 2 :
    tmp = sp.call('clear',shell=True)
    train_images, test_images = dtex.preprocessdata(train_images, train_labels, test_images, class_names)
  elif response == 3 :
    tmp = sp.call('clear',shell=True)
    model = md.modelSetup(train_images, train_labels)  
  elif response == 4 :
    try:
      model
    except NameError:
      print("\nYou need to setup the model!")
      time.sleep(2)
    else:
      tmp = sp.call('clear',shell=True)
      model = md.modelTraining(model, train_images, train_labels)
  elif response == 5 :
    try:
      model
    except NameError:
      print("\nYou need to setup the model!")
      time.sleep(2)
    else:
      tmp = sp.call('clear',shell=True)
      md.aculoss(model, train_images, train_labels, test_images, test_labels)
  elif response == 6 :
    try:
      model
    except NameError:
      print("\nYou need to setup the model!")
      time.sleep(2)
    else:
      tmp = sp.call('clear',shell=True)
      md.predictions(model, test_images, class_names, test_labels)
  elif response == 7 :
    ex = 1
    exit()
  else:
    print("ERROR!")
    time.sleep(2)

# @title MIT License
#
# Copyright (c) 2017 Francois Chollet
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
