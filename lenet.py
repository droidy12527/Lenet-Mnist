#Including All the Libraries needed for Lenet
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Dense, MaxPool2D
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from keras.utils import to_categorical
from keras.datasets import mnist

#Get the data in Train and Test Split
(x_train, y_train) , (x_test, y_test) = mnist.load_data()

#Check the number of Rows and Columns
num_rows = x_train[0].shape[0]
num_cols = x_train[0].shape[1]

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
num_classes = y_test.shape[0]
num_pixels = x_train.shape[1] * x_train.shape[2]





