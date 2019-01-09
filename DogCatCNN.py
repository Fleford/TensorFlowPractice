import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random

DATADIR = r"C:\Users\Fleford Redoloza\PycharmProjects\TensorFlowPractice\kagglecatsanddogs_3367a\PetImages"

CATEGORIES = ["Dog", "Cat"]

img_size = 100

# for category in CATEGORIES:  # do dogs and cats
#     path = os.path.join(DATADIR, category)  # create path to dogs and cats
#     for img in os.listdir(path):  # iterate over each image per dogs and cats
#         img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
#         plt.imshow(img_array, cmap='gray')  # graph it
#         plt.show()  # display!
#
#         img_size = 100
#
#         new_array = cv2.resize(img_array, (img_size, img_size))
#         plt.imshow(new_array, cmap='gray')
#         plt.show()
#
#         break  # we just want one for now so break
#
#     break  # ...and one more!


training_data = []


def create_training_data():
    for category in CATEGORIES:  # do dogs and cats
        path = os.path.join(DATADIR, category)  # create path to dogs and cats
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):  # iterate over each image per dogs and cats
            try:
                img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)  # convert to array
                new_array = cv2.resize(img_array, (img_size, img_size))     # resize image array
                training_data.append([new_array, class_num])
            except Exception as e:
                pass

create_training_data()
random.shuffle(training_data)
print(len(training_data))

for sample in training_data[:10]:
    print(sample[1])

X = []
Y = []

for features, label in training_data:
    X.append(features)
    Y.append(label)
