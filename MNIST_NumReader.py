import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=x_train[0].shape),
  tf.keras.layers.Dense(512, activation=tf.nn.relu),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=3)
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
