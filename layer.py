import tensorflow as tf
from tensorflow.keras.layers import (
    BatchNormalization, ReLU, Dropout, Concatenate, Conv2D, DepthwiseConv2D, GlobalAveragePooling2D,
    Reshape, Dense, Multiply
)
from tensorflow.keras.regularizers import l2

def DenseDDSSPPLayer(inputs, growth_rate, dilation_rate, dropout_rate=0.0, efficient=True):
    """A single DenseDDSSPP Layer with optional efficiency improvements."""
    x = BatchNormalization()(inputs)
    x = ReLU()(x)
    x = Conv2D(filters=growth_rate, kernel_size=1, use_bias=False, kernel_regularizer=l2(1e-4))(x)

    if efficient:
        x = DepthwiseConv2D(kernel_size=3, dilation_rate=dilation_rate, padding='same', use_bias=False)(x)
        x = Conv2D(filters=growth_rate, kernel_size=1, use_bias=False)(x)
    else:
        x = Conv2D(filters=growth_rate, kernel_size=3, dilation_rate=dilation_rate, padding='same', 
                   use_bias=False, kernel_regularizer=l2(1e-4))(x)

    x = BatchNormalization()(x)
    x = ReLU()(x)
    if dropout_rate > 0:
        x = Dropout(rate=dropout_rate)(x)

    return x

def DenseDDSSPP(inputs, growth_rate=48, dropout_rate=0.1):
    """DenseDDSSPP module with multiple dilation rates."""
    x0 = inputs
    x_initial = Conv2D(filters=growth_rate, kernel_size=1, use_bias=False, kernel_regularizer=l2(1e-4))(x0)
    x_initial = BatchNormalization()(x_initial)
    x_initial = ReLU()(x_initial)
    x0 = Concatenate()([x0, x_initial])

    dilation_rates = [6, 12, 24, 36, 48]
    for rate in dilation_rates:
        x = DenseDDSSPPLayer(x0, growth_rate, rate, dropout_rate)
        x0 = Concatenate()([x0, x])
    
    return x0

def se_block(input_tensor, num_filters, ratio=16):
    """Squeeze-and-Excitation Block."""
    se_shape = (1, 1, num_filters)
    se = GlobalAveragePooling2D()(input_tensor)
    se = Reshape(se_shape)(se)
    se = Dense(num_filters // ratio, activation='relu', use_bias=False, kernel_regularizer=l2(1e-4))(se)
    se = Dense(num_filters, activation='sigmoid', use_bias=False, kernel_regularizer=l2(1e-4))(se)
    return Multiply()([input_tensor, se])
