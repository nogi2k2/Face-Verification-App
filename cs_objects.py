from tensorflow.keras.layers import Layer
import tensorflow as tf

class l1_distance(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
