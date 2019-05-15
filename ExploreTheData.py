from __future__ import absolute_import, division, print_function, unicode_literals


def explore(_train_images, _train_labels, _test_images, _test_labels):
    ################################ Train ################################
    print("Quantity of train images, size (px) = ", _train_images.shape)

    print("Quantity of train images: ",len(_train_labels))

    print("Training Vector: ",_train_labels, "\n")
    #######################################################################

    ################################ Test #################################
    print("Quantity of test images, size (px) = ", _test_images.shape)

    print("Quantity of test images: ",len(_test_labels))
    #######################################################################

    exit = raw_input("\nPress any key to Exit! ")
