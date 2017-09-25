# Example code from "Deep Learning with Python"

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
import numpy


# fix random seed
seed = numpy.random.seed(7)

# load pima indians dataset
dataset = numpy.loadtxt("E://!Weiterbildung//!DeepLearning//datasets//pima-indians-diabetes.csv", delimiter=",")

# split into input and output variables
X = dataset[:,0:8]
Y = dataset[:,8]

# define 10-fold cross validation test harness
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

cvscores = []

for train, test in kfold.split(X, Y):

# define NN model with Keras
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu')) # input layer and  first hidden layer
    model.add(Dense(8, activation= 'relu')) # second hidden layer
    model.add(Dense(1, activation= 'sigmoid')) # output layer

# Compile model
    model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])

# Fit the model
    model.fit(X[train], Y[train], validation_split=0.33, epochs=150, batch_size=10, verbose=0)
#model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=150, batch_size=10)

# evaluate the model
    scores = model.evaluate(X[test], Y[test])
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    cvscores.append(scores[1] * 100)
print("%.2f%% (+/- %.2f%%)" % (numpy.mean(cvscores), numpy.std(cvscores)))