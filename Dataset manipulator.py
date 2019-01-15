import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

# mnist = tf.keras.datasets.mnist
#
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
# print(x_train[0])
# x_train, x_test = x_train / 255.0, x_test / 255.0

# Read in saved model (Be sure to have h5py)
new_model = tf.keras.models.load_model("myfirstnumreader.model")
print(new_model.summary())
print(new_model.get_layer(name="dense"))
print(new_model.layers[1])
layer2W, layer2b = new_model.layers[1].get_weights()
print(type(layer2W))
# variables = tf.trainable_variables()
# print(variables[0])
# # Make a prediction!
# index = 5
# predictions = new_model.predict(x_test)
# print(predictions.shape)
# print(np.argmax(predictions[index]))
# plt.imshow(x_test[index], cmap=plt.cm.binary)
# plt.show()
