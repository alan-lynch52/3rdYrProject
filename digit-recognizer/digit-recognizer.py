import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau

train = pd.read_csv("train.csv")
#test = pd.read_csv("test.csv")

#split training into y and x
y = train["label"]
x = train.drop(labels = ["label"],axis = 1)

#print(y.head())
#print(x.head())

del train

#check for missing values
#print(x.isnull().any().describe())
#print(test.isnull().any().describe())

#greyscale the images
x = x / 255.0
#test = test / 255.0

#reshape values
x = x.values.reshape(-1, 28, 28, 1)
#test = test.values.reshape(-1, 28, 28, 1)
#plt.imshow(x[0][:,:,0])
#plt.show()

y = to_categorical(y, num_classes = 10)
random_seed = 2

#split into train and validation set
x_train, x_val, y_train, y_val = train_test_split(x,y, test_size = 0.1, random_state = random_seed)

##plt.imshow(x_train[0][:,:,0])
##plt.show()

cnn = Sequential()
cnn.add(Conv2D(filters = 32, kernel_size = (5,5), padding = 'Same', activation='relu', input_shape = (28,28,1)))
cnn.add(MaxPool2D(pool_size = (2,2), strides = (2,2)))
cnn.add(Dropout(0.25))

cnn.add(Flatten())
cnn.add(Dense(256, activation = 'relu'))
cnn.add(Dropout(0.5))
cnn.add(Dense(10, activation = 'softmax'))

optimizer = RMSprop(lr = 0.001, rho = 0.9, epsilon = 1e-08, decay=0.0)
cnn.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

learning_rate_reduction = ReduceLROnPlateau(
    monitor = 'val_acc',
    patience = 3,
    verbose = 1,
    factor = 0.5,
    min_lr = 0.00001)

epochs = 1
batch_size = 86

#data augmentation
#augmentation techniques:
#grayscale, horizontal & vertical flips, random crops, color jitters, translations, rotations
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.10, # Randomly zoom image 
        width_shift_range=0.10,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.10,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=False,  # randomly flip images
        vertical_flip=False)  # randomly flip images

datagen.fit(x_train)

final_model = cnn.fit_generator(datagen.flow(x_train,y_train, batch_size = batch_size),
                                epochs = epochs, validation_data= (x_val, y_val),
                                verbose = 2, steps_per_epoch = x_train.shape[0] // batch_size,
                                callbacks = [learning_rate_reduction])


##fig, ax = plt.subplots(2,1)
##ax[0].plot(final_model.history['loss'], color='b', label = 'Training Loss')
##ax[0].plot(final_model.history['val_loss'], color='r', label = 'Validation Accuracy')
##legend = ax[0].legend(loc = 'best', shadow = True)
##ax[1].plot(final_model.history['acc'], color='b', label='Training accuracy')
##ax[1].plot(final_model.history['val_acc'], color='r', label='Validation accuracy')
##legend = ax[1].legend(loc = 'best', shadow = True)

#plt.show()
plt.clf()

def plot_c_mtx(c_mtx, classes, cmap=plt.cm.Blues):
    plt.imshow(c_mtx, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    classes = range(10)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = c_mtx.max() / 2.
    for i, j in itertools.product(range(c_mtx.shape[0]), range(c_mtx.shape[1])):
        plt.text(j, i, c_mtx[i,j],
            horizontalalignment = "center",
            color = "white" if c_mtx[i,j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

pred = cnn.predict(x_val)
pred_classes = np.argmax(pred, axis = 1)
y_true = np.argmax(y_val, axis = 1)
c_mtx = confusion_matrix(y_true, pred_classes)
plot_c_mtx(c_mtx, range(10))
plt.show()
#heat_map = sns.heatmap(c_mtx, annot=True, fmt='d')



