from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Input , 
    Conv2D , 
    Dropout , 
    MaxPooling2D , 
    Conv2DTranspose , 
    concatenate , 
    Lambda
)

image_dim = 128
image_channels = 1

inputs = Input((image_dim , image_dim , image_channels))
func = Lambda(lambda x: x / 255)(inputs)

conv_1 = Conv2D(64 , (3 , 3) , activation = 'relu' , kernel_initializer = 'he_normal' , padding='same')(func)
conv_1 = Dropout(0.1)(conv_1)
conv_1 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_1)

conv_2 = Conv2D(64 , (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_1)
conv_2 = Dropout(0.1)(conv_2)
conv_2 = Conv2D(64 , (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_2)

pool_2 = MaxPooling2D((2, 2))(conv_2)

conv_3 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_2)
conv_3 = Dropout(0.2)(conv_3)
conv_3 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_3)

pool_3 = MaxPooling2D((2, 2))(conv_3)

conv_4 = Conv2D(16 , (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_3)
conv_4 = Dropout(0.2)(conv_4)
conv_4 = Conv2D(16 , (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_4)

pool_4 = MaxPooling2D(pool_size=(2, 2))(conv_4)

conv_5 = Conv2D(8 , (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(pool_4)
conv_5 = Dropout(0.3)(conv_5)
conv_5 = Conv2D(8 , (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_5)

union_6 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(conv_5)
union_6 = concatenate([union_6, conv_4])

conv_6 = Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(union_6)
conv_6 = Dropout(0.2)(conv_6)
conv_6 = Conv2D(8, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_6)

union_7 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(conv_6)
union_7 = concatenate([union_7, conv_3])

conv_7 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(union_7)
conv_7 = Dropout(0.2)(conv_7)
conv_7 = Conv2D(16, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_7)

union_8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv_7)
union_8 = concatenate([union_8, conv_2])

conv_8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(union_8)
conv_8 = Dropout(0.1)(conv_8)
conv_8 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(conv_8)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(conv_8)

v_net = tf.keras.Model(inputs=[inputs], outputs=[outputs])
v_net.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

v_net.fit(X_train , Y_train , epochs = 100)
