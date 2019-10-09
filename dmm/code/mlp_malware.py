import os
#os.environ["THEANO_FLAGS"] = "device=gpu0,floatX=float32"
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout,  Activation
import numpy as np
from keras.utils import np_utils
from scipy import io 

np.random.seed(1234)

def perf_measure(y_true, y_pred):
    TP_FN = np.count_nonzero(y_true)
    FP_TN = y_true.shape[0]  - TP_FN
    FN = np.where((y_true - y_pred) == 1)[0].shape[0]
    TP = TP_FN - FN
    FP = np.count_nonzero(y_pred) - TP
    TN = FP_TN - FP
    Precision = float(float(TP) / float(TP + FP + 1e-9))
    Recall = float(float(TP) / float((TP + FN + 1e-9)))
    accuracy = float(float(TP + TN) / float((TP_FN + FP_TN + 1e-9)))
    F1 =  2*((Precision * Recall) / (Precision + Recall))
    return Precision, Recall, accuracy, F1

# Loading the training and testing samples.
train = np.load('../data/train_pdf.npz')
X_train = train['X_train']
y_train = train['Y_train']

test = np.load('../data/test_pdf.npz')
X_test = test['X_test']
y_test = test['Y_test']

Y_train = np_utils.to_categorical(y_train)
Y_test = np_utils.to_categorical(y_test)

print X_train.shape
print X_test.shape
print Y_train.shape
print Y_test.shape

"""
# Training the PDF classifier.

batch_size = 100 
epochs = 16

model = Sequential()
model.add(Dense(100, activation='sigmoid', input_shape=(135,)))
model.add(Dropout(0.2))
model.add(Dense(50, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='sigmoid'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer= 'adam',
              metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs,
          verbose=1, validation_data=(X_test, Y_test))
model.save('pdf_mlp.h5')
"""

# Loading and testing the well-trained model.
model = load_model('../model/pdf_mlp.h5')
model.summary()

print('evaluating test data....')
P_test = model.predict_classes(X_test, verbose = 0)
(precision, recall, accuracy, F1) = perf_measure(y_true=y_test, y_pred=P_test)
print("Precision: %s Recall: %s Accuracy: %s F1: %s" %(precision, recall, accuracy, F1))

# Collecting the training set for explanation.
PP = model.predict(X_train)
PP =  PP[:,1]
PP = PP.reshape(4999, 1)
print PP.shape
print X_train.shape
io.savemat('../data/data_GMM', {'X': X_train, 'y':PP})
