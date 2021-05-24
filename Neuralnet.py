import numpy as np
from numpy import argmax
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
# define example
X_train = [[1, 3, 2, 0, 3, 2], [2, 1, 3, 0, 3, 2], [1, 1, 3, 0, 5, 1], [2, 1, 3, 0, 3, 2]]
X_train = np.array(X_train)
y_train =[[1, 3, 2, 0],[1, 3, 2, 0]]
y_train = np.array(y_train)
# one hot encode
encoded = np.array(np.append(to_categorical(X_train[0][0:4]), [X_train[0][4], X_train[0][5]]), dtype=object).reshape(1, 18)

for i in range(X_train.shape[0] - 1):
    encoded = np.append(encoded, np.append(to_categorical(X_train[i + 1][0:4]), [X_train[i + 1][4], X_train[i + 1][5]]).reshape(1, 18), axis=0)

# print(encoded)
# print(encoded.shape)
X_train = encoded
encoded = np.array(to_categorical(y_train[0])).reshape(1, 16)
# print(encoded)
# print(encoded.shape)
# print(X_train)

for i in range(y_train.shape[0] - 1):
    encoded = np.append(encoded, to_categorical(y_train[i + 1]).reshape(1, 16), axis=0)

y_train = encoded
# print(y_train)

model = Sequential()
model.add(Dense(90, activation='relu', input_dim=X_train.shape[1]))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(y_train.shape[1], activation="softmax"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics="accuracy")

model.fit(X_train, y_train, epochs=1024)

# print(encoded)
# # invert encoding
# inverted = argmax(encoded[0])
# print(inverted)