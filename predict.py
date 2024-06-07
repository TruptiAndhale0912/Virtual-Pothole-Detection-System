# predict.py

import tensorflow as tf
import numpy as np
from keras_preprocessing.image import load_img, img_to_array

# Load the model
classifierLoad = tf.keras.models.load_model('model.h5')

def predict_herb(img_path):
    test_image = load_img(img_path, target_size=(200, 200))
    test_image = img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    test_image /= 255.0

    result = classifierLoad.predict(test_image)
    flag = True if result[0][1] > result[0][0] else False
    msg = (
        "Thanks for your awareness. Potholes location and details noted. We will take action soon. Thanks."
        if flag else
        "Thanks for your awareness. But as per our criteria provided image does not contain any potholes."
    )
    return msg, flag
