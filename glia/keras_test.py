from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD

model = Sequential()

model.add(Dense(output_dim=64, input_dim=100, init="glorot_uniform"))
model.add(Activation("relu"))
model.add(Dense(output_dim=10, init="glorot_uniform"))
model.add(Activation("softmax"))

model.compile(loss='categorical_crossentropy', optimizer='sgd')
