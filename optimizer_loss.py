import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

#Section1 - Import data

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

print("<--------------------------Section1----------------------->")
print("Section 1: Import Data")
print("Train images shape: {0}".format(train_images.shape))
print("Train labels shape: {0}".format(train_labels.shape))
print("Test images shape: {0}".format(test_images.shape))
print("Test labels shape: {0}".format(test_labels.shape))
# print("Display sameple data: {0}".format(train_images[0]))

#Display images

from matplotlib import pyplot

for i in range(9):
	# define subplot
	pyplot.subplot(330 + 1 + i)
	# plot raw pixel data
	pyplot.imshow(train_images[i], cmap=pyplot.get_cmap('gray'))
# show the figure
pyplot.show()

#Section 2 - Reshaping and Data Preparation

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype("float32") / 255

from tensorflow.keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

print("Section 2: Reshaping and Data Preparation (Normalisation to 0 and 1)")
print("Train images shape: {0}".format(train_images.shape))
print("Train labels shape: {0}".format(train_labels.shape))
print("Test images shape: {0}".format(test_images.shape))
print("Test labels shape: {0}".format(test_labels.shape))
# print("Display sameple data: {0}".format(train_images[0]))

from sklearn.model_selection import train_test_split
print()
print("Data after splitting to training and validation set")
X_train, X_val, y_train, y_val = train_test_split(train_images, train_labels, test_size=0.2)
print("Train images shape: {0}".format(X_train.shape))
print("Train labels shape: {0}".format(y_train.shape))
print("Validation images shape: {0}".format(X_val.shape))
print("Validation labels shape: {0}".format(y_val.shape))

#Section 3 - Training Model
from tensorflow.keras import layers
from tensorflow.keras import regularizers
input_shape = (28, 28, 1)

model = keras.Sequential()
model.add(layers.InputLayer(input_shape=input_shape))
model.add(layers.BatchNormalization())
model.add(layers.Conv2D(32,(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(rate=0.5))
model.add(layers.Conv2D(64,(3, 3), padding="same", activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(rate=0.5))
model.add(layers.Flatten())
model.add(layers.Dense(64, 
    activation="relu"))
model.add(layers.Dense(10, activation="softmax"))

print("Section 3: Model Training")
print("Model Summary: ")
print(model.summary())

#RMSProp - Model 1 with categorical_crossentropy
opt = tf.keras.optimizers.RMSprop(learning_rate=0.0008)
model.compile(optimizer = opt,
             loss = 'categorical_crossentropy',
              metrics=['accuracy']
             )

callbacks = [
    keras.callbacks.ModelCheckpoint("A3_model1.keras")
]

history = model.fit(X_train, y_train, epochs=30, batch_size=64, callbacks=callbacks, validation_data=(X_val, y_val))

print("Model Evaluation of RMSProp with categorical cross-entropy")
model = keras.models.load_model(
    "A3_model1.keras")
print("Evaluation Test accuracy score of RMSProp with categorical cross-entropy: {0}".format(model.evaluate(test_images, test_labels)[1]))


print("Plot model 1:   ")
loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.title("Training Loss of RMSProp with categorical cross-entropy")
plt.legend()
plt.show()

#RMSProp - Model 2 with CategoricalHinge
opt = tf.keras.optimizers.RMSprop()
model.compile(optimizer = opt,
             loss = 'CategoricalHinge',
              metrics=['accuracy']
             )

callbacks = [
    keras.callbacks.ModelCheckpoint("A3_model2.keras")
]

history = model.fit(X_train, y_train, epochs=30, batch_size=64, callbacks=callbacks, validation_data=(X_val, y_val))

print("Model Evaluation of RMSProp with categorical hinge")
model = keras.models.load_model(
    "A3_model2.keras")
print("Evaluation Test accuracy score of RMSProp with categorical hinge: {0}".format(model.evaluate(test_images, test_labels)[1]))

print("Plot model 2:   ")
loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.title("Training Loss of RMSProp with categorical hinge")
plt.legend()
plt.show()

#RMSProp - Model 3 with KLDivergence
opt = tf.keras.optimizers.RMSprop()
model.compile(optimizer = opt,
             loss = 'KLDivergence',
              metrics=['accuracy']
             )

callbacks = [
    keras.callbacks.ModelCheckpoint("A3_model3.keras")
]

history = model.fit(X_train, y_train, epochs=30, batch_size=64, callbacks=callbacks, validation_data=(X_val, y_val))

print("Model Evaluation of RMSProp with KL-Divergence")
model = keras.models.load_model(
    "A3_model3.keras")
print("Evaluation Test accuracy score of RMSProp with KL-Divergence: {0}".format(model.evaluate(test_images, test_labels)[1]))

print("Plot model 3:   ")
loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.title("Training Loss of of RMSProp with KL-Divergence")
plt.legend()
plt.show()

#Adagrad - Model 1 with categorical_crossentropy
opt = keras.optimizers.Adagrad(learning_rate = 0.017)
model.compile(optimizer = opt,
             loss = 'categorical_crossentropy',
              metrics=['accuracy']
             )

callbacks = [
    keras.callbacks.ModelCheckpoint("Adagrad_categorical_crossentropy.keras")
]

history = model.fit(X_train, y_train, epochs=30, batch_size=64, callbacks=callbacks, validation_data=(X_val, y_val))

print("Model Evaluation of Adagrad with Categorical crossentropy")
model = keras.models.load_model(
    "Adagrad_categorical_crossentropy.keras")
print("Evaluation Test accuracy score of Adagrad with Categorical crossentropy: {0}".format(model.evaluate(test_images, test_labels)[1]))

print("Plot model Adagrad with Categorical crossentropy:   ")
loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.title("Training Loss of of Adagrad with Categorical crossentropy")
plt.legend()
plt.show()

#Adagrad - Model 2 with CategoricalHinge
opt = keras.optimizers.Adagrad(learning_rate = 0.017)
model.compile(optimizer = opt,
             loss = 'CategoricalHinge',
              metrics=['accuracy']
             )

callbacks = [
    keras.callbacks.ModelCheckpoint("Adagrad_categorical_hinge.keras")
]

history = model.fit(X_train, y_train, epochs=30, batch_size=64, callbacks=callbacks, validation_data=(X_val, y_val))

print("Model Evaluation of Adagrad with categorical hinge")
model = keras.models.load_model(
    "Adagrad_categorical_hinge.keras")
print("Evaluation Test accuracy score of Adagrad with categorical hinge: {0}".format(model.evaluate(test_images, test_labels)[1]))

print("Plot model Adagrad with categorical hinge:   ")
loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.title("Training Loss of of Adagrad with categorical hinge")
plt.legend()
plt.show()

#Adagrad - Model 3 with KLDivergence
opt = keras.optimizers.Adagrad(learning_rate = 0.017)
model.compile(optimizer = opt,
             loss = 'KLDivergence',
              metrics=['accuracy']
             )

callbacks = [
    keras.callbacks.ModelCheckpoint("Adagrad_KlDivergence.keras")
]

history = model.fit(X_train, y_train, epochs=30, batch_size=64, callbacks=callbacks, validation_data=(X_val, y_val))

print("Model Evaluation of Adagrad with KL-Divergence")
model = keras.models.load_model(
    "Adagrad_KlDivergence.keras")
print("Evaluation Test accuracy score of Adagrad with KL-Divergence: {0}".format(model.evaluate(test_images, test_labels)[1]))

print("Plot model Adagrad with KL-Divergence:   ")
loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.title("Training Loss of of Adagrad with KL-Divergence")
plt.legend()
plt.show()

"""
Created on Mon May 30 19:12:06 2022

@author: teres
"""
#SGD,categorical_crossentropy learning rate 0.01 with momentum0.9
opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer = opt,
             loss = 'categorical_crossentropy',
              metrics=['accuracy']
             )
callbacks = [
    keras.callbacks.ModelCheckpoint("sgd_cc_0.01_0.9.keras")
]

history = model.fit(X_train, y_train, epochs=30, batch_size=64, callbacks=callbacks, validation_data=(X_val, y_val))
print("Model Evaluation")
model = keras.models.load_model(
    "sgd_cc_0.01_0.9.keras")
print("Evaluation Test accuracy score of SGD with categorical corss-entropy: {0}".format(model.evaluate(test_images, test_labels)[1]))

print("Plot model SGD with Categorical crossentropy:   ")
loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.title("Training Loss of of SGD with Categorical crossentropy")
plt.legend()
plt.show()



#SGD,categorical_hinge learning rate 0.01 with momentum0.9

opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer = opt,
             loss = 'categorical_hinge',
              metrics=['accuracy']
             )
callbacks = [
    keras.callbacks.ModelCheckpoint("sgd_ch_0.01_0.9.keras")
]

history = model.fit(X_train, y_train, epochs=30, batch_size=64, callbacks=callbacks, validation_data=(X_val, y_val))
print("Model Evaluation")
model = keras.models.load_model(
    "sgd_ch_0.01_0.9.keras")
print("Evaluation Test accuracy score of SGD with categorical hinge: {0}".format(model.evaluate(test_images, test_labels)[1]))

print("Plot model SGD with Categorical Hinge:   ")
loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.title("Training Loss of of SGD with Categorical Hinge")
plt.legend()
plt.show()



#SGD,kl_divergence learning rate 0.01 with momentum0.9

opt = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
model.compile(optimizer = opt,
             loss = 'kl_divergence',
              metrics=['accuracy']
             )
callbacks = [
    keras.callbacks.ModelCheckpoint("sgd_kl_0.01_0.9.keras")
]

history = model.fit(X_train, y_train, epochs=30, batch_size=64, callbacks=callbacks, validation_data=(X_val, y_val))
print("Model Evaluation")
model = keras.models.load_model(
    "sgd_kl_0.01_0.9.keras")
print("Evaluation Test accuracy score of SGD with KL Divergence: {0}".format(model.evaluate(test_images, test_labels)[1]))

print("Plot model SGD with KL-Divergence:   ")
loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.title("Training Loss of of SGD with KL-Divergence")
plt.legend()
plt.show()

#ADAM - Model 1 with Categorical-Crossentropy
opt=tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer = opt,
             loss = 'categorical_crossentropy',
              metrics=['accuracy']
             )

callbacks = [
    keras.callbacks.ModelCheckpoint("A3_Adam_model1.keras")
]

history = model.fit(X_train, y_train, epochs=30, batch_size=64, callbacks=callbacks, validation_data=(X_val, y_val))

print("Model Evaluation of ADAM with categorical cross-entropy")
model = keras.models.load_model(
    "A3_Adam_model1.keras")
print("Evaluation Test accuracy score of ADAM with categorical cross-entropy: {0}".format(model.evaluate(test_images, test_labels)[1]))

import matplotlib.pyplot as plt
print("Plot model 1:   ")
loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.title("Training Loss of ADAM with categorical cross-entropy")
plt.legend()
plt.show()

#ADAM - Model 2 with categorical_hinge
opt=tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer = opt,
             loss = 'CategoricalHinge',
              metrics=['accuracy']
             )

callbacks = [
    keras.callbacks.ModelCheckpoint("A3_Adam_model2.keras")
]

history = model.fit(X_train, y_train, epochs=30, batch_size=64, callbacks=callbacks, validation_data=(X_val, y_val))

print("Model Evaluation of ADAM with categorical hinge")
model = keras.models.load_model(
    "A3_Adam_model2.keras")
print("Evaluation Test accuracy score of ADAM with categorical hinge: {0}".format(model.evaluate(test_images, test_labels)[1]))

import matplotlib.pyplot as plt
print("Plot model 1:   ")
loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.title("Training Loss of ADAM with categorical hinge")
plt.legend()
plt.show()

#ADAM - Model 3 with KLDivergence
opt=tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(optimizer = opt,
             loss = 'KLDivergence',
              metrics=['accuracy']
             )

callbacks = [
    keras.callbacks.ModelCheckpoint("A3_Adam_model3.keras")
]

history = model.fit(X_train, y_train, epochs=30, batch_size=64, callbacks=callbacks, validation_data=(X_val, y_val))

print("Model Evaluation of ADAM with KL-Divergence")
model = keras.models.load_model(
    "A3_Adam_model3.keras")
print("Evaluation Test accuracy score of ADAM with KL-Divergence: {0}".format(model.evaluate(test_images, test_labels)[1]))

import matplotlib.pyplot as plt
print("Plot model 1:   ")
loss = history.history["loss"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training Loss")
plt.title("Training Loss of ADAM with KL-Divergence")
plt.legend()
plt.show()