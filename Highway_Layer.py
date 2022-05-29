import keras
from keras import backend as K
from keras.layers import Dense, Activation, Multiply, Add, Lambda
from keras import initializers

def highway_layer(value, activation="tanh", transform_gate_bias=-1.0):
    dim = K.int_shape(value)[-1]
    transform_gate_bias_initializer = keras.initializers.Constant(transform_gate_bias)
    transform_gate = Dense(units=dim, bias_initializer=transform_gate_bias_initializer)(value)
    transform_gate = Activation("sigmoid")(transform_gate)
    carry_gate = Lambda(lambda x: 1.0 - x, output_shape=(dim,))(transform_gate)
    transformed_data = Dense(units=dim)(value)
    transformed_data = Activation(activation)(transformed_data)
    transformed_gated = Multiply()([transform_gate, transformed_data])
    identity_gated = Multiply()([carry_gate, value])
    value = Add()([transformed_gated, identity_gated])
    return value