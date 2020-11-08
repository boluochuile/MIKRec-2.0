import tensorflow as tf
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam
import numpy as np

class MyModel(tf.keras.models.Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.dense1 = tf.keras.layers.Dense(10, activation=tf.nn.relu)
        self.dense2 = tf.keras.layers.Dense(5, activation=tf.nn.softmax)

    def call(self, inputs, training=True):
        x = self.dense1(inputs)
        return self.dense2(x)

model = MyModel()
model.compile(loss=binary_crossentropy, optimizer=Adam(learning_rate=0.001), metrics=[tf.keras.metrics.MSE])

x = np.random.uniform(-1, 1, size=(4,10))
y = np.random.uniform(0, 1, size=(4,5))
print(x)
print(y)

model.fit(
    x,
    y,
    epochs=10
)
x2 = np.random.uniform(-1, 1, size=(2,10))
print(model.predict(x))