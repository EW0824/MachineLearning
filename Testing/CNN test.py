
import tensorflow as tf
from tensorflow import keras

import numpy as np

print(np.__version__)
print(tf.__version__)
from sklearn.datasets import load_sample_image

model = keras.applications.resnet50.ResNet50(weights="imagenet")




# Load sample images
china = load_sample_image("china.jpg") / 255
flower = load_sample_image("flower.jpg") / 255
images = np.array([china, flower])



images_resized = tf.image.resize(images, [224, 224])
inputs = keras.applications.resnet50.preprocess_input(images_resized * 255)


y_proba = model.predict(inputs)

