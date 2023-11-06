import tensorflow as tf
from tensorflow.keras import layers, models

# Custom Capsule Layer
class CapsuleLayer(layers.Layer):
    def __init__(self, num_capsules, dim_capsule, routings=3, **kwargs):
        super(CapsuleLayer, self).__init__(**kwargs)
        self.num_capsules = num_capsules
        self.dim_capsule = dim_capsule
        self.routings = routings

    def build(self, input_shape):
        input_dim_capsule = input_shape[-1]
        self.W = self.add_weight(shape=[input_dim_capsule, self.num_capsules * self.dim_capsule],
                                initializer='glorot_uniform',
                                name='W')

    def call(self, inputs):
        # Expand dimensions to handle routing iterations
        inputs_expand = tf.expand_dims(inputs, axis=2)
        inputs_tiled = tf.tile(inputs_expand, [1, 1, self.num_capsules, 1])

        # Perform matrix multiplication between W and inputs
        input_hat = tf.matmul(inputs_tiled, self.W)

        # Routing algorithm
        b = tf.zeros(shape=[tf.shape(inputs)[0], tf.shape(inputs)[1], self.num_capsules])
        for i in range(self.routings):
            c = tf.nn.softmax(b, axis=2)
            outputs = self.squash(tf.reduce_sum(c[..., tf.newaxis] * input_hat, axis=1, keepdims=True))

            if i < self.routings - 1:
                b += tf.reduce_sum(input_hat * outputs, axis=-1)

        return tf.squeeze(outputs, axis=1)

    def squash(self, vector):
        squared_norm = tf.reduce_sum(tf.square(vector), axis=-1, keepdims=True)
        safe_norm = tf.sqrt(squared_norm + tf.keras.backend.epsilon())
        squash_factor = squared_norm / (1. + squared_norm)
        unit_vector = vector / safe_norm
        return squash_factor * unit_vector

# Building the Capsule Network Model
def CapsNet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)
    
    x = layers.Conv2D(32, kernel_size=3, activation='relu')(inputs)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Conv2D(64, kernel_size=3, activation='relu')(x)
    x = layers.MaxPooling2D(pool_size=2)(x)
    x = layers.Flatten()(x)

    # Capsule Layer
    x = layers.Reshape((-1, 64))(x)
    output = CapsuleLayer(num_capsules=num_classes, dim_capsule=16, routings=3)(x)

    # Final Classification
#     x = layers.Lambda(lambda x: tf.sqrt(tf.reduce_sum(tf.square(x), axis=-1)), name='norm')(x)
    # outputs = layers.Activation('softmax')(x)

    return models.Model(inputs, output)

# Example usage:
# input_shape = (128, 128, 3)  # Replace with your image input shape
# num_classes = 2  # Number of classes for classification

# caps_cnn_model = build_caps_cnn(input_shape, num_classes)
# caps_cnn_model.summary()
