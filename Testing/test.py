import numpy as np
import tensorflow as tf


from sklearn.datasets import load_iris

#
# ### Aman's code to enable the GPU
# #from tensorflow.python.compiler.mlcompute import mlcompute
# #tf.compat.v1.disable_eager_execution()
# #mlcompute.set_mlc_device(device_name='gpu')
# #print("is_apple_mlc_enabled %s" % mlcompute.is_apple_mlc_enabled())
# #print("is_tf_compiled_with_apple_mlc %s" % #mlcompute.is_tf_compiled_with_apple_mlc())
# #print(f"eagerly? {tf.executing_eagerly()}")
# #print(tf.config.list_logical_devices())

x = np.random.random((10000, 5))
y = np.random.random((10000, 2))

x2 = np.random.random((2000, 5))
y2 = np.random.random((2000, 2))

x3 = np.random.random((2000, 5))
y3 = np.random.random((2000, 2))

inp = tf.keras.layers.Input(shape = (5,))
l1 = tf.keras.layers.Dense(256, activation = 'sigmoid')(inp)
l1 = tf.keras.layers.Dense(256, activation = 'sigmoid')(l1)
l1 = tf.keras.layers.Dense(256, activation = 'sigmoid')(l1)
l1 = tf.keras.layers.Dense(256, activation = 'sigmoid')(l1)
l1 = tf.keras.layers.Dense(256, activation = 'sigmoid')(l1)
o = tf.keras.layers.Dense(2, activation = 'sigmoid')(l1)


model = tf.keras.models.Model(inputs = [inp], outputs = [o])
model.compile(optimizer = "Adam", loss = "mse")



model.fit(x, y, validation_data = (x2, y2), batch_size = 50, epochs = 50)

print(f"\bEvaluation: {model.evaluate(x3, y3)}")

print(model.predict([[0.37423738, 0.92195177, 0.52481983, 0.85074282, 0.5976232]]))
