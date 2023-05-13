alex = Sequential()

alex.add(Conv2D(96 , kernel_size = (11 , 11) , strides = 4 , activation = "relu" , input_shape = (227  , 227 , 3)))
alex.add(MaxPooling2D(pool_size = (3 , 3) , strides = 2))

alex.add(Conv2D(256 , kernel_size = (5 , 5) , padding = "same" , activation = "relu"))
alex.add(MaxPooling2D(pool_size = (3 , 3) , strides = 2))

alex.add(Conv2D(384 , kernel_size = (3 , 3) , padding = "same" , activation = "relu"))
alex.add(Conv2D(384 , kernel_size = (3 , 3) , padding = "same" , activation = "relu"))
alex.add(Conv2D(256 , kernel_size = (3 , 3) , padding = "same" , activation = "relu"))
alex.add(MaxPooling2D(pool_size = (3 , 3) , strides = 2))

alex.add(Flatten())

alex.add(Dropout(rate = 0.5))
         
alex.add(Dense(4096 , activation = "relu"))
alex.add(Dropout(rate = 0.5))

alex.add(Dense(4096 , activation = "relu"))
alex.add(Dense(4096 , activation = "relu"))

alex.add(Dense(1 , activation = "sigmoid"))
