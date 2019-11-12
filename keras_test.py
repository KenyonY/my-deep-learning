import os 
print(os.cpu_count)

# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# os.environ['CUDA_VISIBLE_ORDER']='PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES']='-1'

import tensorflow as tf
from tensorflow.keras.layers import Dense, Concatenate, Conv2D,Conv1D,Convolution1D,\
Flatten, BatchNormalization,Input
from tensorflow.keras.models import Sequential, Model
N = 1000
x_input = Input(shape=(N, 1))
x = Flatten()(x_input)
x = Dense(10, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

disc = Model(x_input, x, name='discriminator')
disc.summary()