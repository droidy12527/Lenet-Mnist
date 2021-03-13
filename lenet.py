#Including All the Libraries needed for Lenet
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D, Flatten
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.optimizers import Adadelta
from keras.utils import to_categorical
from keras.datasets import mnist
import datetime

#Get the data in Train and Test Split
(x_train, y_train) , (x_test, y_test) = mnist.load_data()

#Check the number of Rows and Columns
num_rows = x_train[0].shape[0]
num_cols = x_train[1].shape[0]

#Get the data in Shape , By adding 1 at end that denotes that it is Black and White and not BGR which represents 3
x_train = x_train.reshape(x_train.shape[0], num_rows, num_cols, 1)
x_test = x_test.reshape(x_test.shape[0], num_rows, num_cols, 1)

#Lets define a variable with the Input Shape
img_shape = (num_rows, num_cols, 1)

#Lets Normalise the data ---> Current Image Pixel Values Range from 0-255 normalise them to 0-1
#For That first change the type to float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

#Normalise the Values now
x_train /= 255
x_test /= 255

#Lets Check the Y part
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

#Lets declare the number of classes
num_classes = y_test.shape[1] 
num_pixels = x_train.shape[1] * x_train.shape[2]

#Lets Define A Clone Model of Lenet
model = Sequential()

model.add(Conv2D(20, (5,5), padding="same", input_shape=img_shape, activation="relu"))
model.add(MaxPool2D(pool_size=(2,2) ,strides=(2,2)))

model.add(Conv2D(50, (5,5), padding="same", activation="relu"))
model.add(MaxPool2D(pool_size=(2,2) ,strides=(2,2)))

model.add(Flatten())
model.add(Dense(500, activation="relu"))
model.add(Dense(num_classes, activation="softmax"))

print(model.summary())

#Compile The model
model.compile(loss="categorical_crossentropy", optimizer=Adadelta(), metrics=['accuracy'])

#Define Batch Size and Epochs
batch_size = 128
epochs = 40

#Defining Callback for Model

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

checkpoint = ModelCheckpoint("./lenet_handwritten.h5", monitor="val_loss", mode="min", save_best_only = True, verbose=1)
earlystop = EarlyStopping(monitor = 'val_loss', min_delta = 0, patience = 3, verbose = 1, restore_best_weights = True)
tensorboard = TensorBoard(log_dir=log_dir, histogram_freq=1)

callbacks = [checkpoint, earlystop, tensorboard]

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, callbacks=callbacks, validation_data=(x_test, y_test), shuffle=True)


