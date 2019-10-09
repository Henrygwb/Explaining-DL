import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import to_categorical
from scipy import io
import gzip
np.random.seed(1234)

def spit_data(data, i):
    p = model.predict(data)
    Y = p[:,i]
    data = data.reshape(data.shape[0], 784)
#    a = np.where((Y<0.9))
#    Y = np.delete(Y, a[0],axis = 0)
#    data = np.delete(data, a[0], axis = 0)
    print data.shape
    print Y.shape
#    print np.where(Y<0.9)
    #print Y
    idx = np.where(Y ==1)
    Y[idx] = 0.999999
    Y = np.log(Y) - np.log(1-Y)
    name = '../data/data_'+str(i)
    io.savemat(name, {'X':data, 'Y':Y})
    return name

def dimensional_reduction(name, k = 300):
    X = io.loadmat(name)['X']
    Y = io.loadmat(name)['Y']
    print X.shape
    print Y.shape
    var = []
    for i in xrange(X.shape[1]):
        var_tmp = np.var(X[:,i])
        var.append(var_tmp)
    var = np.asarray(var)
    sort_var = np.argsort(var)
    k_1 = 784 - k
    print k_1
    var_select = sort_var[0:k_1, ]
    X_300 = np.delete(X, var_select, axis = 1)
    print X_300.shape
    print Y.shape
    name_1 = name+'_'+str(k)
    io.savemat(name_1, {'X': X_300, 'Y':Y, 'var_idx': sort_var[k_1:784, ]})

def load_data(kind):
    labels_path = os.path.join('%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join('%s-images-idx3-ubyte.gz' % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
         labels = np.frombuffer(lbpath.read(), dtype = np.uint8, offset = 8)
    with gzip.open(images_path, 'rb') as imgpath:
         images = np.frombuffer(imgpath.read(), dtype = np.uint8, offset = 16).reshape(len(labels), 784)
    return labels, images

print 'Loading data ...'
train_labels, train_images = load_data('../data/train')
test_labels, test_images = load_data('../data/test')

X_train = train_images.astype('float32')
X_train = X_train/255
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = test_images.astype('float32')
X_test = X_test/255
X_test = X_test.reshape(10000, 28, 28, 1)

Y_train = to_categorical(train_labels, num_classes = 10)
Y_test = to_categorical(test_labels, num_classes = 10)

num_classes = 10

"""
batch_size = 1000
epochs = 30

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5, 5), strides = 1, padding = 'same', activation='relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Conv2D(filters = 64, kernel_size = (5, 5), strides = 1, padding = 'same', activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=['accuracy'])

history = model.fit(X_train, Y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
model.save('fashion_mnist_cnn.h5')
print('Test loss:', score[0])
print('Test accuracy:', score[1])
"""

model = load_model('../model/fashion_mnist_cnn.h5')
model.summary()

## select the sample belonging to the explaining class and prepare the dataset for fitting dmm-men
class_exp = 1

data_all = np.concatenate((X_train, X_test))
label_all = np.concatenate((Y_train, Y_test))

data_exp = []

for i in xrange(data_all.shape[0]):
    if label_all[i, class_exp] == 1:
            data_exp.append(data_all[i,:])

data_exp = np.asarray((data_exp))
print data_exp.shape

Y_exp = np.zeros((data_exp.shape[0],10))
Y_exp[:, class_exp]=1

print '*********************test_legitimate*****************'
score = model.evaluate(X_test, Y_test, batch_size=10000)
print score[1]

score = model.evaluate(data_exp, Y_exp, verbose=0)
print score[1]

file_name = spit_data(data_exp, class_exp)
dimensional_reduction(file_name)