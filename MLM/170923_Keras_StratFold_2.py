# Example code from "Deep Learning with Python"

from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
import numpy

# fix random seed
seed = numpy.random.seed(7)

def create_model():
    # define NN model with Keras
    model = Sequential()
    model.add(Dense(12, input_dim=8, activation='relu'))  # input layer and  first hidden layer
    model.add(Dense(8, activation='relu'))  # second hidden layer
    model.add(Dense(1, activation='sigmoid'))  # output layer

    # Compile model
    model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
    return model

# load pima indians dataset
dataset = numpy.loadtxt("E://!Weiterbildung//!DeepLearning//datasets//pima-indians-diabetes.csv", delimiter=",")

# split into input and output variables
X = dataset[:,0:8]
Y = dataset[:,8]

# create model
model = KerasClassifier(build_fn = create_model, epochs=150, batch_size=10, verbose=0)

# evaluate using 10-fold cross validation
kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
results = cross_val_score(model, X, Y, cv=kfold)
print(results.mean())
