from flask import Flask
from PIL import Image, ImageOps
from numpy import asarray
from Yolo.yolo_detection_images import detectWifiRouter
import numpy as np
import librosa
#from keras.models import load_model
import tensorflow as tf

app = Flask(__name__)


@app.route('/')
def hello():
    return 'hello'

@app.route('/predict-image', methods=['POST','GET'])
def detect():
    # img=request.args['image']
    # img_path='images/'+img
    # results=detectWifiRouter(img_path)
    # return jsonify(results)
    if (request.method == 'POST'):
        if request.files:
            image = Image.open(request.files['image'])
            numpydata = asarray(image)
            result = detectWifiRouter(numpydata)
            return result

        return "file not saved"

if __name__ == '__main__':
    app.run()
