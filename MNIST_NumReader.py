import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# x1 = tf.constant(5)
# x2 = tf.constant(6)
#
# result = tf.multiply(x1, x2)
# print(result)
#
# with tf.Session() as sess:
#     output = sess.run(result)
# print(output)

print(tf.__version__)

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Defining the model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(128, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(10, activation=tf.nn.softmax))

# Training the model
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(x_train, y_train, epochs=3)

# Check performance
val_loss, val_acc = model.evaluate(x_test, y_test)
print(val_loss, val_acc)

# Save Model
model.save("myfirstnumreader.model")

# Read in saved model (Be sure to have h5py)
new_model = tf.keras.models.load_model("myfirstnumreader.model")

# Make a prediction!
index = 5
predictions = new_model.predict([x_test])
print(np.argmax(predictions[index]))
plt.imshow(x_test[index], cmap=plt.cm.binary)
plt.show()
