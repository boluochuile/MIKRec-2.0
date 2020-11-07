from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding
import numpy as np
import tensorflow as tf

model = Sequential()
model.add(Embedding(3, 2, input_length=7))

print(model.layers[0].get_weights())

model.compile('rmsprop', 'mse')
data = np.array([[0,1,2,1,1,0,1],[0,1,2,1,1,0,1]])
print(model.predict(data))
