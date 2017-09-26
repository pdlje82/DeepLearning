# MLP for Pima Indians Dataset Serialize to JSON and HDF5
from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
from keras.models import model_from_yaml
import numpy
import os

# fix random seed for reproducibility
numpy.random.seed(7)
# load pima indians dataset
dataset = numpy.loadtxt("E://!Weiterbildung//!DeepLearning//datasets//pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = Sequential()
model.add(Dense(12, input_dim=8, kernel_initializer= 'uniform' , activation= 'relu' ))
model.add(Dense(8, kernel_initializer= 'uniform' , activation= 'relu' ))
model.add(Dense(1, kernel_initializer= 'uniform' , activation= 'sigmoid' ))
# Compile model
model.compile(loss= 'binary_crossentropy' , optimizer= 'adam' , metrics=[ 'accuracy' ])
# Fit the model
model.fit(X, Y, epochs=150, batch_size=10, verbose=0)
# evaluate the model
scores = model.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))


# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
print("Saved JSON model to disk")

# serialize model to YAML
model_yaml = model.to_yaml()
with open("model.yaml", "w") as yaml_file:
    yaml_file.write(model_yaml)
print("Saved YAML to disk")

# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved weights to disk")

# later...
# load json and create model
json_file = open( 'model.json' , 'r' )
loaded_model_json = json_file.read()
json_file.close()
loaded_model1 = model_from_json(loaded_model_json)
print("Loaded JSON model from disk")

# load YAML and create model
yaml_file = open( 'model.yaml' , 'r' )
loaded_model_yaml = yaml_file.read()
yaml_file.close()
loaded_model2 = model_from_yaml(loaded_model_yaml)
print("Loaded YAML model from disk")

# load weights into new model
loaded_model1.load_weights("model.h5")
print("Loaded weights from disk")

# evaluate loaded model on test data
loaded_model1.compile(loss= 'binary_crossentropy' , optimizer= 'rmsprop' , metrics=[ 'accuracy' ])
score = loaded_model1.evaluate(X, Y, verbose=0)
print("%s: %.2f%%" % (loaded_model1.metrics_names[1], score[1]*100))
