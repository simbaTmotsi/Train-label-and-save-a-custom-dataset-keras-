#Simbarashe Timothy Motsi
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import model_from_json

from keras import backend as k
k.set_image_dim_ordering('tf')
#to hide warnings
#os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from keras.utils import np_utils
from keras.models import Sequential 
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop, adam

img_rows = 128
img_cols = 128
num_channel = 1
img_data_list = []

PATH = os.getcwd()
# Define data path
data_path = PATH + '/data'
data_dir_list = os.listdir(data_path)

for dataset in data_dir_list:
    img_list = os.listdir(data_path+'/'+dataset)
    print ('\n Loaded the images of dataset -'+'{}\n'.format(dataset))

    for img in img_list:
        input_img = cv2.imread(data_path+'/'+dataset+'/'+img)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
        input_img_resize = cv2.resize(input_img, (128,128))
        img_data_list.append(input_img_resize)  
        
img_data = np.array(img_data_list)
img_data = img_data.astype('float32')
img_data /= 255
print(img_data.shape)

#importing the backend keras 2 format
{
 "image_data_format": "channel_last",
 "epsilon":1e-07,
 "floatx": "float32",
 "backend":"tensorflow"
 }

if num_channel == 1:
    if k.image_dim_ordering()=='th':
        img_data = np.expand_dims(img_data,axis = 1)
        print(img_data.shape)
    else:
        img_data = np.expand_dims(img_data, axis = 4)
        print(img_data.shape)
else:
    if k.image_dim_ordering()=='th':
        img_data = np.rollaxis(img_data,3,1)
        print(img_data.shape)

#assigning labels        
num_classes = 31
num_of_samples = img_data.shape[0]
labels = np.ones((num_of_samples,),dtype='int64')

labels[0:32]=0          #1 - 33 samples
labels[33:67]=1         #2 - 34 samples
labels[68:100]=2        #3 - 33 samples
labels[101:134]=3       #4 - 33 samples
labels[135:168]=4       #5 - 33 samples
labels[169:560]=5       #A - 400 samples
labels[561:1050]=6      #B - 400 samples
labels[1051:1450]=7     #C - 400 samples
labels[1451:1850]=8     #D - 400 samples
labels[1851:2250]=9     #E - 400 samples
labels[2251:2650]=10    #F - 400 samples
labels[2651:3050]=11    #G - 400 samples
labels[3051:3450]=12    #H - 400 samples
labels[3451:3850]=13    #I - 400 samples
labels[3851:4250]=14    #J - 400 samples
labels[4251:4650]=15    #K - 400 samples
labels[4651:5050]=16    #L - 400 samples
labels[5051:5450]=17    #M - 400 samples
labels[5451:6250]=18    #N - 400 samples
labels[6251:6650]=19    #O - 400 samples
labels[6651:7050]=20    #P - 400 samples
labels[7051:7450]=21    #Q - 400 samples
labels[7451:7850]=22    #R - 400 samples
labels[7851:7950]=23    #S - 100 samples
labels[8051:8150]=24    #T - 100 samples
labels[8151:8250]=25    #U - 100 samples
labels[8251:8350]=26    #V - 100 samples
labels[8351:8450]=27    #W - 100 samples
labels[8451:8550]=28    #X - 100 samples
labels[8551:8650]=29    #Y - 100 samples
labels[8651:]=30        #Z - 34 samples

names = ['1','2','3','4','5',
         'A','B','C','D','E','F',
         'G','H','I','J','K','L',
         'M','N','O','P','Q','R',
         'S','T','U','V','W','X','Y','Z']

#converting class labels to one-hot encoding
Y = np_utils.to_categorical(labels,num_classes)

#shufflling the data
x,y = shuffle(img_data, Y, random_state=2)

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

model = Sequential()
input_shape=img_data[0].shape
model.add(Conv2D(32, (3,3),padding='same',input_shape=input_shape))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',optimizer='adadelta',metrics=["accuracy"])
#keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#model.compile(loss='categorical_crossentropy', optimizer='Adam',metrics=["accuracy"])
print("......About to start training......\n")


#sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd,metrics=["accuracy"])
#model.compile(loss='categorical_crossentropy', optimizer='rmsprop',metrics=["accuracy"])

# Viewing model_configuration

model.summary()
model.get_config()
model.layers[0].get_config()
model.layers[0].input_shape            
model.layers[0].output_shape            
model.layers[0].get_weights()
np.shape(model.layers[0].get_weights()[0])
model.layers[0].trainable

#%%
# Training
#setting the iteration parameters
hist = model.fit(X_train, y_train, batch_size=32, epochs=100,verbose=1, validation_split=0.2)

# Training with callbacks
from keras import callbacks

filename='model_train_new.csv'
csv_log=callbacks.CSVLogger(filename, separator=',', append=False)

early_stopping=callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='min')

filepath="Best-weights-my_model-{epoch:03d}-{loss:.4f}-{acc:.4f}.hdf5"

checkpoint = callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=True, mode='min')

callbacks_list = [csv_log,early_stopping,checkpoint]

#graph
num_epoch = 100

# visualizing losses and accuracy
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['acc']
val_acc=hist.history['val_acc']
xc=range(num_epoch)

plt.figure(1,figsize=(7,5))
plt.plot(xc,train_loss)
plt.plot(xc,val_loss)
plt.xlabel('num of Epochs')
plt.ylabel('loss')
plt.title('train_loss vs val_loss')
plt.grid(True)
plt.legend(['train','val'])
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])

plt.figure(2,figsize=(7,5))
plt.plot(xc,train_acc)
plt.plot(xc,val_acc)
plt.xlabel('num of Epochs')
plt.ylabel('accuracy')
plt.title('train_acc vs val_acc')
plt.grid(True)
plt.legend(['train','val'],loc=4)
#print plt.style.available # use bmh, classic,ggplot for big pictures
plt.style.use(['classic'])
plt.show()

print("....training complete...\n")
#saving the model and weights
model_json = model.to_json()
with open("Sign.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("Sign.h5")
print("Saved model to disk")

"""
#Test image
test_image = cv2.imread('test/T_15.jpg')
test_image = cv2.cvtColor(test_image,cv2.COLOR_BGR2GRAY)
test_image = cv2.resize(test_image,(128,128))
test_image = test_image.astype('float32')
test_image /= 255

if num_channel == 1:
    if k.image_dim_ordering()=='th':
        test_image = np.expand_dims(test_image, axis = 0)
        test_image = np.expand_dims(test_image, axis = 0)
        print(test_image)
        
    else:
        test_image = np.expand_dims(test_image, axis = 3)
        test_image = np.expand_dims(test_image, axis = 0)
        print(test_image.shape)
        
else:
    if k.image_dim_ordering=='th':
        test_image = np.rollaxis(test_image, 2,0)
        test_image = np.expand_dims(test_image, axis = 0)
        print(test_image.shape)
        
print((model.predict(test_image)))
print(model.predict_classes(test_image))
"""
print(".....COMPLETE....")
