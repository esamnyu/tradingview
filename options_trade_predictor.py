import numpy as np
from keras.models import Sequential
from keras.layers import Dense

# Assuming X_train and y_train are your training data,
# where X_train is an array with each row being a different trade
# characterized by features such as [entering price, exiting price, entering time, exiting time, entering volatility, exiting volatility, etc.]
# and y_train is a binary array indicating whether each trade was good (1) or not (0)

model = Sequential()
model.add(Dense(32, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))  # Sigmoid function for binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=10)
