import os
from flask import Flask, request, jsonify
import base64
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

app = Flask(__name__)
APP_ROOT = os.path.dirname(os.path.abspath(__file__))

# Load the trained model
model_path = os.path.join(APP_ROOT, 'model.h5')
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found at {model_path}")
model = load_model(model_path)

def prepare_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(200, 200))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0
        return img_array
    except Exception as e:
        print(f"Error in prepare_image: {e}")
        raise

def predict_pothole(img_path):
    try:
        img = prepare_image(img_path)
        result = model.predict(img)
        if result[0][0] > result[0][1]:
            msg = "Thanks for your awareness. But as per our criteria provided image does not contain any potholes."
            flag = False
        else:
            msg = "Thanks for your awareness. Potholes location and details noted. We will take action soon. Thanks."
            flag = True
        return msg, flag
    except Exception as e:
        print(f"Error in predict_pothole: {e}")
        raise

@app.route('/analysis', methods=['POST'])
def analysis():
    try:
        imageData = request.form['imageData']
        imgdata = base64.b64decode(imageData)
        path1 = os.path.join('static', 'test_images')
        if not os.path.exists(path1):
            os.makedirs(path1)
        filename = os.path.join(path1, "test.jpg")
        with open(filename, 'wb') as f:
            f.write(imgdata)
        
        msg, flag = predict_pothole(filename)
        response = {"info": msg, "flag": flag}
        return jsonify(response)
    except Exception as e:
        print(f"Error in /analysis route: {e}")
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(port=5000, host='0.0.0.0', debug=True)
