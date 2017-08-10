import tensorflow as tf
from tensorflow.contrib import keras
from keras.models import Sequential
from keras.layers import Dense, InputLayer

def make_ann(initial, n_hidden, w_hidden, n_Out=1, out_activ='linear', return_model=False):
    '''
    use Keras to quickly construct a simple ANN for function approximation

    hidden layers have ReLU activation; final layer is linear

    initial:    input tensor 
    n_hidden:    number of hidden layers
    w_hidden:    width (# of units) of hidden layers
    n_out:       number of output units
    out_activ:   activation on output units

    returns:    output tensor (and model if returnModel is True)
    '''
    model = Sequential()
    model.add(InputLayer(input_tensor=tf.expand_dims(initial, -1)))
    for _ in range(n_hidden):
        model.add(Dense(w_hidden, activation='relu'))
    model.add(Dense(n_out, activation=out_activ))

    output = tf.squeeze(model.get_layer(index=-1).output)
    if returnModel:
        return output, model
    else:
        return output 
