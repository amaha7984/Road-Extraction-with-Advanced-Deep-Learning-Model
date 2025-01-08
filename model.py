import tensorflow as tf
from tensorflow.keras.layers import Input, UpSampling2D, Conv2D, BatchNormalization, Activation, Concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.applications import Xception
from layers import DenseDDSSPP, se_block

def DeepLabV3PlusDenseDDSSPP(input_shape=(512, 512, 3)):
    """DeepLabV3+ with DenseDDSSPP integration."""
    inputs = Input(input_shape)

    base_model = Xception(weights='imagenet', include_top=False, input_tensor=inputs)
    image_features = base_model.get_layer('block13_sepconv2_bn').output

    x_a = DenseDDSSPP(image_features)
    x_a = UpSampling2D((4, 4), interpolation="bilinear")(x_a)

    x_b = base_model.get_layer('block3_sepconv2').output
    x_b = Conv2D(filters=48, kernel_size=1, padding='same', use_bias=False)(x_b)
    x_b = BatchNormalization()(x_b)
    x_b = Activation('relu')(x_b)
    x_b = tf.image.resize(x_b, [128, 128])

    x = Concatenate()([x_a, x_b])
    x = se_block(x, num_filters=1360) #num of filters vary with different layer selection

    for _ in range(4):  # Repeating Conv2D-BatchNorm-Activation layers
        x = Conv2D(filters=256, kernel_size=3, padding='same', activation='relu', use_bias=False)(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

    x = UpSampling2D((4, 4), interpolation="bilinear")(x)
    x = Conv2D(1, (1, 1), name='output_layer')(x)
    x = Activation('sigmoid')(x)

    model = Model(inputs=inputs, outputs=x)
    return model
