import tensorflow as tf
from tensorflow.keras import backend as K
import tensorflow.keras as keras


# Define the custom loss function and use the exact name 'my_package>mse_with_positive_pressure'
def mse_with_positive_pressure(y_true, y_pred):
    mse = K.mean(K.square(y_pred - y_true), axis=-1)
    positive_pressure = K.maximum(0.0, -y_pred) ** 2
    return mse + K.mean(positive_pressure)

# Map the custom loss function with the specific prefix used in the model
custom_objects = {'my_package>mse_with_positive_pressure': mse_with_positive_pressure}

# Load the model with the custom_objects parameter
model = keras.models.load_model('mooot.h5', custom_objects=custom_objects)
model.load_weights('models/ckpt_10.weights.h5')
