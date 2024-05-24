# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import re
import shutil
import string
import tensorflow as tf
import onnxmltools

from tensorflow.keras import layers
from tensorflow.keras import losses
from zipfile import ZipFile

"""Read dataset."""

with ZipFile("drive/MyDrive/toxicity.zip", "r") as zObject:
    zObject.extractall(
        path=".")

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 64
embedding_dim = 16
seed = 42

raw_train_ds = tf.keras.utils.text_dataset_from_directory(
  "toxicity",
  batch_size=batch_size,
  validation_split=0.2,
  subset="training",
  seed=seed)

class_names = raw_train_ds.class_names

raw_val_ds = tf.keras.utils.text_dataset_from_directory(
  "toxicity",
  batch_size=batch_size,
  validation_split=0.2,
  subset="validation",
  seed=seed)


raw_test_ds = tf.keras.utils.text_dataset_from_directory(
  "toxicity",
  batch_size=batch_size)

"""Init vocabulary and tokenizer."""

def custom_standardization(input_data):
  lowercase = tf.strings.lower(input_data)
  stripped_html = tf.strings.regex_replace(lowercase, '<br />', ' ')
  return tf.strings.regex_replace(stripped_html,
                                  '[%s]' % re.escape(string.punctuation),
                                  '')

max_features = 100_000
sequence_length = 250

vectorize_layer = layers.TextVectorization(
    standardize=custom_standardization,
    max_tokens=max_features,
    output_mode='int',
    output_sequence_length=sequence_length)

# Make a text-only dataset (without labels), then call adapt
train_text = raw_train_ds.map(lambda x, y: x)
vectorize_layer.adapt(train_text)

def vectorize_text(text, label):
  text = tf.expand_dims(text, -1)
  return vectorize_layer(text), label

train_ds = raw_train_ds.map(vectorize_text)
val_ds = raw_val_ds.map(vectorize_text)
test_ds = raw_test_ds.map(vectorize_text)

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

model = tf.keras.Sequential([
  layers.Input(batch_shape=(1, sequence_length)),
  layers.Embedding(input_dim=max_features, output_dim=embedding_dim),
  layers.Dropout(0.2),
  layers.GlobalAveragePooling1D(),
  layers.Dropout(0.2),
  layers.Dense(64, activation="relu"),
  layers.Dropout(0.2),
  layers.Dense(len(class_names), activation="sigmoid")])

"""Train model."""

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
epochs = 1

model.compile(optimizer=optimizer,
              loss=loss,
              metrics=["accuracy"])

history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs,
  batch_size=batch_size)

export_model = tf.keras.Sequential([
  vectorize_layer,
  model,
  layers.Activation("sigmoid")
])

export_model.compile(
  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False), optimizer="adam", metrics=["accuracy"]
)

"""Export TF model to ONNX one."""

onnx_model = onnxmltools.convert_keras(model, target_opset=18)
onnxmltools.utils.save_model(onnx_model, "superego.onnx")