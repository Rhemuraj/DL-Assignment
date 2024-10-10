from keras.optimizers import Adam
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.layers import BatchNormalization, GlobalAveragePooling2D, Input, Add
from keras.models import Model

# Define convolution with batch normalization
def Conv2d_BN(x, nb_filter, kernel_size, padding='same', strides=(1,1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x
  
# Define Residual Block for ResNet34 (2 convolution layers)
def Residual_Block(input_model, nb_filter, kernel_size, strides=(1,1), with_conv_shortcut=False):
    x = Conv2d_BN(input_model, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    
    if with_conv_shortcut:
        shortcut = Conv2d_BN(input_model, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = Add()([x, shortcut])
    else:
        x = Add()([x, input_model])
    
    return x
    
# Build ResNet34
def ResNet34(width, height, depth, classes):
    Img = Input(shape=(width, height, depth))
    
    x = Conv2d_BN(Img, 64, (7,7), strides=(2,2), padding='same')
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same')(x)  

    x = Residual_Block(x, nb_filter=64, kernel_size=(3,3))
    x = Residual_Block(x, nb_filter=64, kernel_size=(3,3))
    x = Residual_Block(x, nb_filter=64, kernel_size=(3,3))
    
    x = Residual_Block(x, nb_filter=128, kernel_size=(3,3), strides=(2,2), with_conv_shortcut=True)
    x = Residual_Block(x, nb_filter=128, kernel_size=(3,3))
    x = Residual_Block(x, nb_filter=128, kernel_size=(3,3))
    x = Residual_Block(x, nb_filter=128, kernel_size=(3,3))
    
    x = Residual_Block(x, nb_filter=256, kernel_size=(3,3), strides=(2,2), with_conv_shortcut=True)
    x = Residual_Block(x, nb_filter=256, kernel_size=(3,3))
    x = Residual_Block(x, nb_filter=256, kernel_size=(3,3))
    x = Residual_Block(x, nb_filter=256, kernel_size=(3,3))
    x = Residual_Block(x, nb_filter=256, kernel_size=(3,3))
    x = Residual_Block(x, nb_filter=256, kernel_size=(3,3))
    
    x = Residual_Block(x, nb_filter=512, kernel_size=(3,3), strides=(2,2), with_conv_shortcut=True)
    x = Residual_Block(x, nb_filter=512, kernel_size=(3,3))
    x = Residual_Block(x, nb_filter=512, kernel_size=(3,3))

    x = GlobalAveragePooling2D()(x)
    x = Dense(classes, activation='softmax')(x)
    
    model = Model(inputs=Img, outputs=x)
    return model  

ResNet34_model = ResNet34(224, 224, 3, 2)

ResNet34_model.summary()
ResNet34_model.compile(optimizer=Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                       loss='categorical_crossentropy', metrics=['accuracy'])

model = ResNet34_model

