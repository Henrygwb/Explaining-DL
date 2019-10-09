import os
os.environ["THEANO_FLAGS"] = "module=FAST_RUN,device=gpu,floatX=float32"
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import to_categorical
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Flatten
import theano
import theano.tensor as T
from scipy import io
import scipy as sp
import scipy.optimize
from numpy import linalg as LA
import gzip
np.random.seed(1234)

def load_data(kind):
    labels_path = os.path.join('%s-labels-idx1-ubyte.gz' % kind)
    images_path = os.path.join('%s-images-idx3-ubyte.gz' % kind)
    with gzip.open(labels_path, 'rb') as lbpath:
         labels = np.frombuffer(lbpath.read(), dtype = np.uint8, offset = 8)
    with gzip.open(images_path, 'rb') as imgpath:
         images = np.frombuffer(imgpath.read(), dtype = np.uint8, offset = 16).reshape(len(labels), 784)
    return labels, images

print 'Loading data ...'
train_labels, train_images = load_data('train')
test_labels, test_images = load_data('test')


X_train = train_images.astype('float32')
X_train = X_train/255
X_train = X_train.reshape(60000, 28, 28, 1)
X_test = test_images.astype('float32')
X_test = X_test/255
X_test = X_test.reshape(10000, 28, 28, 1)

Y_train = to_categorical(train_labels, num_classes = 10)
Y_test = to_categorical(test_labels, num_classes = 10)


batch_size = 1000
num_classes = 10
epochs = 30
"""
model = Sequential()
#Conv2D(filters, kernel_size, strides=(1, 1), padding='valid', data_format=None, dilation_rate=(1, 1), activation=None, use_bias=True, kernel_initializer='glorot_uniform', bias_initializer='zeros', kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None, kernel_constraint=None, bias_constraint=None)
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


data_all = np.concatenate((X_train, X_test))
label_all = np.concatenate((Y_train, Y_test))

data_0 = []
data_1 = []
data_2 = []
data_3 = []
data_4 = []

data_5 = []
data_6 = []
data_7 = []
data_8 = []
data_9 = []
for i in xrange(data_all.shape[0]):
        if label_all[i,0] == 1:
                data_0.append(data_all[i,:])
        elif label_all[i, 1]==1:
                data_1.append(data_all[i,:])
        elif label_all[i, 2]==1:
                data_2.append(data_all[i,:])    
        elif label_all[i, 3]==1:
                data_3.append(data_all[i,:])    
        elif label_all[i, 4]==1:
                data_4.append(data_all[i,:])
        elif label_all[i, 5]==1:
                data_5.append(data_all[i,:])
        elif label_all[i, 6]==1:
                data_6.append(data_all[i,:])
        elif label_all[i, 7]==1:
                data_7.append(data_all[i,:])
        elif label_all[i, 8]==1:        
                data_8.append(data_all[i,:])    
        elif label_all[i, 9]==1:        
                data_9.append(data_all[i,:])
"""
model = load_model('fashion_mnist_cnn.h5')
model.summary()
print '*********************test_legitimate*****************'
score = model.evaluate(X_test, Y_test, batch_size=10000)
print score[1]

data_0 = np.asarray((data_0))
data_1 = np.asarray((data_1))
data_2 = np.asarray((data_2))
data_3 = np.asarray((data_3))
data_4 = np.asarray((data_4))
data_5 = np.asarray((data_5))
data_6 = np.asarray((data_6))
data_7 = np.asarray((data_7))
data_8 = np.asarray((data_8))
data_9 = np.asarray((data_9))

print data_0.shape
print data_1.shape
print data_2.shape
print data_3.shape
print data_4.shape
print data_5.shape
print data_6.shape
print data_7.shape
print data_8.shape
print data_9.shape

Y_0 = np.zeros((data_0.shape[0],10))
Y_0[:,0]=1

Y_1 = np.zeros((data_1.shape[0],10))
Y_1[:,1]=1

Y_2 = np.zeros((data_2.shape[0],10))
Y_2[:,2]=1

Y_3 = np.zeros((data_3.shape[0],10))
Y_3[:,3]=1

Y_4 = np.zeros((data_4.shape[0],10))
Y_4[:,4]=1

Y_5 = np.zeros((data_5.shape[0],10))
Y_5[:,5]=1

Y_6 = np.zeros((data_6.shape[0],10))
Y_6[:,6]=1

Y_7 = np.zeros((data_7.shape[0],10))
Y_7[:,7]=1

Y_8 = np.zeros((data_8.shape[0],10))
Y_8[:,8]=1

Y_9 = np.zeros((data_9.shape[0],10))
Y_9[:,9]=1

score = model.evaluate(data_0, Y_0)
print score[1]

score = model.evaluate(data_1, Y_1)
print score[1]

score = model.evaluate(data_2, Y_2)
print score[1]

score = model.evaluate(data_3, Y_3)
print score[1]

score = model.evaluate(data_4, Y_4)
print score[1]

score = model.evaluate(data_5, Y_5)
print score[1]

score = model.evaluate(data_6, Y_6)
print score[1]

score = model.evaluate(data_7, Y_7)
print score[1]

score = model.evaluate(data_8, Y_8)
print score[1]

score = model.evaluate(data_9, Y_9)
print score[1]

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
    name = 'data_'+str(i)
    io.savemat(name, {'X':data, 'Y':Y})

spit_data(data_0, 0)
spit_data(data_1, 1)
spit_data(data_2, 2)
spit_data(data_3, 3)
spit_data(data_4, 4)
spit_data(data_5, 5)
spit_data(data_6, 6)
spit_data(data_7, 7)
spit_data(data_8, 8)
spit_data(data_9, 9)

