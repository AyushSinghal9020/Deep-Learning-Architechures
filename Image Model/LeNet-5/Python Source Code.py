import matplotlib.pyplot as plt
from tensorflow import keras
from keras.datasets import mnist
from keras import Sequential 
from keras.layers import Conv2D , AveragePooling2D , Flatten , Dense 

(X_train , y_train) , (X_test , y_test) = mnist.load_data()

model.add(Conv2D(6 , kernel_size = (5, 5) , padding = "valid" , activation = "tanh" , input_shape = (28  , 28 , 1)))
model.add(AveragePooling2D(pool_size = (2 , 2) , strides = 2 , padding = "valid"))

model.add(Conv2D(16 , kernel_size = (5 , 5) , padding = "valid" , activation = "tanh"))
model.add(AveragePooling2D(pool_size = (2 , 2) , strides = 2 , padding = "valid"))

model.add(Flatten())

model.add(Dense(120 , activation = "tanh"))
model.add(Dense(84 , activation = "tanh"))
model.add(Dense(10 , activation = "softmax"))

model.compile(optimizer = "adam" , loss = "sparse_categorical_crossentropy" , metrics = ["accuracy"])

history = model.fit(X_train , y_train , epochs = 100)

score = model.evaluate(X_test, y_test, verbose = 0)
print('Test loss: {}%'.format(score[0] * 100))
print('Test score: {}%'.format(score[1] * 100))

print("MLP Error: %.2f%%" % (100 - score[1] * 100))
