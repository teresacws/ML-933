# -*- coding: utf-8 -*-
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
print("Evaluation Test accuracy score: {0}".format(model.evaluate(test_images, test_labels)[1]))



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
print("Evaluation Test accuracy score: {0}".format(model.evaluate(test_images, test_labels)[1]))



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
print("Evaluation Test accuracy score: {0}".format(model.evaluate(test_images, test_labels)[1]))