# Example code from "Deep Learning with Python"

from keras.models import Sequential
from keras.layers import Dense
import numpy


# fix random seed
numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("E://!Weiterbildung//!DeepLearning//datasets//pima-indians-diabetes.csv", delimiter=",")

# split into input and output variables
X = dataset[:,0:8]
Y = dataset[:,8]

# define NN model with Keras
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) # input layer and  first hidden layer
model.add(Dense(8, activation= 'relu')) # second hidden layer
model.add(Dense(1, activation= 'sigmoid')) # output layer

# Compile model
model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])

# Fit the model
model.fit(X, Y, epochs=150, batch_size=10)

# evaluate the model
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))