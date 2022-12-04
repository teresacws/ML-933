#!/usr/bin/env python
# coding: utf-8

# In[1]:
import argparse

parser = argparse.ArgumentParser(description='TEMPERAFURE FORECASTING USING TRANSFORMER')
parser.add_argument('-d',metavar='dataset',required=True,dest='dataset', action='store',help='path and name of dataset')
args = parser.parse_args()

import os
fname = os.path.join("./jena_climate_2009_2016.csv")
with open(fname) as f:
    data = f.read()
lines = data.split("\n")
header = lines[0].split(",")
lines = lines[1:]
print(header)
print(len(lines))


# In[2]:


import numpy as np
temperature = np.zeros((len(lines),))
raw_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(",")[1:]]
    temperature[i] = values[1]
    raw_data[i, :] = values[:]


# In[3]:


from matplotlib import pyplot as plt
#plt.plot(range(len(temperature)), temperature)

#Calucaltion of the number of samples
num_train_samples = int(0.5 * len(raw_data))
num_val_samples = int(0.25 * len(raw_data))
num_test_samples = len(raw_data) - num_train_samples - num_val_samples
print("num_train_samples:", num_train_samples)
print("num_val_samples:", num_val_samples)
print("num_test_samples:", num_test_samples)


# In[6]:


#normalization
mean = raw_data[:num_train_samples].mean(axis=0)
raw_data -= mean
std = raw_data[:num_train_samples].std(axis=0)
raw_data /= std


# In[7]:


#splitting dataset into training, validation and testing dataset
import numpy as np
import tensorflow as tf
from tensorflow import keras

sampling_rate = 6
sequence_length = 120
delay = sampling_rate * (sequence_length + 24 - 1)
batch_size = 256

train_dataset = keras.preprocessing.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=0,
    end_index=num_train_samples)

val_dataset = keras.preprocessing.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples,
    end_index=num_train_samples + num_val_samples)

test_dataset = keras.preprocessing.timeseries_dataset_from_array(
    raw_data[:-delay],
    targets=temperature[delay:],
    sampling_rate=sampling_rate,
    sequence_length=sequence_length,
    shuffle=True,
    batch_size=batch_size,
    start_index=num_train_samples + num_val_samples)


# In[8]:


#Outputting the shape of the samples and targets in the training dataset
for samples, targets in train_dataset:
    print("samples shape:", samples.shape)
    print("targets shape:", targets.shape)
    break


# In[9]:
# Transformer
from tensorflow.keras import layers
#Attention and Normalization
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
x = layers.MultiHeadAttention(key_dim=256,num_heads=4, dropout=0.25)(inputs, inputs)
x = layers.Dropout(0.25)(x)
x = layers.LayerNormalization(epsilon=1e-6)(x)
res = x + inputs

#Feed Forward Part
x = layers.Conv1D(filters=4, kernel_size=1, activation="relu")(res)
x = layers.Dropout(0.25)(x)
x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
x = layers.LayerNormalization(epsilon=1e-6)(x)
x = x + res


outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
              loss="mse",
              metrics=["mae"])

callbacks = [
keras.callbacks.ModelCheckpoint("jena_transformer_encoder.keras",
save_best_only=True)
]
history = model.fit(train_dataset,
                    epochs=50,
                    validation_data=val_dataset,
                    callbacks=callbacks)

    
model = keras.models.load_model("jena_transformer_encoder.keras")
print(f"Test acc: {model.evaluate(test_dataset)[1]:.2f}")

print("Single Head Attention:")
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))

#Multiple enconder layers

for _ in range(4):
    # Attention and Normalization
    x = layers.MultiHeadAttention(key_dim=256, num_heads=1, dropout=0.25)(inputs, inputs)
    #Dropout
    x = layers.Dropout(0.25)(x)
    #Layer Normalization
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

# Feed Forward Part
    x = layers.Conv1D(filters=4, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = x+res
x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
for dim in ([128]):
    x = layers.Dense(dim, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
    
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
callbacks = [keras.callbacks.ModelCheckpoint("jena_transformer_head1.keras",save_best_only=True)]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=20,
                    validation_data=val_dataset,
                    callbacks=callbacks)
model = keras.models.load_model("jena_transformer_head1.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")


# In[ ]:


import matplotlib.pyplot as plt
loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Validation MAE")
plt.title("Training and validation MAE")
plt.legend()
plt.show()


# In[ ]:


print("Double Head Attention:")
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
#Multiple enconder layers
for _ in range(4):
# Attention and Normalization
    x = layers.MultiHeadAttention(key_dim=256, num_heads=2, dropout=0.25)(inputs, inputs)
    x = layers.Dropout(0.25)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

# Feed Forward Part
    x = layers.Conv1D(filters=4, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = x+res
x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
for dim in ([128]):
    x = layers.Dense(dim, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
callbacks = [keras.callbacks.ModelCheckpoint("jena_transformer_head2.keras",save_best_only=True)]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=20,
                    validation_data=val_dataset,
                    callbacks=callbacks)
model = keras.models.load_model("jena_transformer_head2.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")


# In[ ]:


loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Validation MAE")
plt.title("Training and validation MAE")
plt.legend()
plt.show()


# In[ ]:


print("4-head Attention:")
inputs = keras.Input(shape=(sequence_length, raw_data.shape[-1]))
#Multiple enconder layers
for _ in range(4):
# Attention and Normalization
    x = layers.MultiHeadAttention(key_dim=256, num_heads=4, dropout=0.25)(inputs, inputs)
    x = layers.Dropout(0.25)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs

# Feed Forward Part
    x = layers.Conv1D(filters=4, kernel_size=1, activation="relu")(res)
    x = layers.Dropout(0.25)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = x+res
x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
for dim in ([128]):
    x = layers.Dense(dim, activation="relu")(x)
    x = layers.Dropout(0.4)(x)
outputs = layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
callbacks = [keras.callbacks.ModelCheckpoint("jena_transformer_head4.keras",save_best_only=True)]
model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
history = model.fit(train_dataset,
                    epochs=20,
                    validation_data=val_dataset,
                    callbacks=callbacks)
model = keras.models.load_model("jena_transformer_head4.keras")
print(f"Test MAE: {model.evaluate(test_dataset)[1]:.2f}")


# In[ ]:


loss = history.history["mae"]
val_loss = history.history["val_mae"]
epochs = range(1, len(loss) + 1)
plt.figure()
plt.plot(epochs, loss, "bo", label="Training MAE")
plt.plot(epochs, val_loss, "b", label="Validation MAE")
plt.title("Training and validation MAE")
plt.legend()
plt.show()

