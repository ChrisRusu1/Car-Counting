from __future__ import absolute_import, division, print_function, unicode_literals
import wget
import tensorflow as tf
#import tensornets as nets
import cv2
import numpy as np
import time
import os
from tensorflow import keras
cifar10 = tf.keras.datasets.cifar10
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0
def create_model():
    model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
    ])
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    return model

checkpoint_path = "training_Car/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

model = create_model()
model.summary()

model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))
model.save_weights(checkpoint_path.format(epoch=10))

test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print ("Loss: " + str(test_loss) + "\nAccuracy: " + str(test_accuracy*100))
print(test_labels[:100])

latest = tf.train.latest_checkpoint(checkpoint_dir)
model.load_weights(latest)
model.load_weights(latest)
loss, acc = model.evaluate(test_images, test_labels)
print("Restored model, accuracy: {:5.2f}%".format(100*acc))
