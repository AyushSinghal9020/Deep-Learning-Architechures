zf = Sequential()

zf.add(Conv2D(96 , kernel_size = (7 , 7) , strides = 2 , activation = "relu" , input_shape = (224  , 224 , 3)))
zf.add(MaxPooling2D(pool_size = (3 , 3) , strides = 2))

zf.add(Conv2D(256 , kernel_size = (3 , 3) , activation = "relu"))
zf.add(MaxPooling2D(pool_size = (3 , 3) , strides = 2))

zf.add(Conv2D(384 , kernel_size = (3 , 3) , padding = "same" , activation = "relu"))
zf.add(Conv2D(384 , kernel_size = (3 , 3) , padding = "same" , activation = "relu"))
zf.add(Conv2D(256 , kernel_size = (3 , 3) , padding = "same" , activation = "relu"))

zf.add(MaxPooling2D(pool_size = (3 , 3) , strides = 2))

zf.add(Flatten())

zf.add(Dense(4096 , activation = "relu"))
zf.add(Dense(4096 , activation = "relu"))

zf.add(Dense(1 , activation = "sigmoid"))
