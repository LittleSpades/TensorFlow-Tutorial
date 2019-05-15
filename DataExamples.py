from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def preprocessdata(_train_images, _train_labels, _test_images, _class_names):
    ######################### Pre-Processing Data ########################
    
    response = input("What Training Image do you want to use as an example? (1-60,000) ")
    
    plt.figure()
    plt.imshow(_train_images[response]) #Training Image 1
    plt.colorbar()  
    plt.grid(False)
    plt.savefig('Pre-Processed Data Test.png')
    print("Created 'Pre-Processed Data Test.png'\n")

    #The values must be in a range of 0 to 1 before feeding to the neural network model
    _train_images = _train_images / 255.0
    _test_images = _test_images / 255.0

    plt.figure()
    plt.imshow(_train_images[response]) #Training Image 1
    plt.colorbar()  
    plt.grid(False)
    plt.savefig('Pre-Processed Training Data.png')
    print("Created 'Pre-Processed Training Data.png'\n")
    #######################################################################


    ####################### Training Data Example #########################
    plt.figure(figsize=(10,10))

    for i in range(25):

        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(_train_images[i], cmap=plt.cm.binary)
        plt.xlabel(_class_names[_train_labels[i]])

    plt.savefig('Training Data Set Example.png')
    print("Created 'Training Data Set Example.png'\n")
    #######################################################################

    exit = raw_input("\nPress any key to Exit! ")

    return _train_images, _test_images