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


@app.route('/predict-audio',methods=['POST','GET'])
def prediction():
    if request.method=='GET':
        if len(ans)!=0:
            return str(ans[-1])
        else:
            return "send your audio format"
    if request.method=='POST':
        if request.files:
            f=request.files.get('file')
            x, sr = librosa.load(f, res_type='kaiser_fast')
            mfccs = np.mean(librosa.feature.mfcc(y=x, sr=sr, n_mfcc=40).T, axis=0)
            mfccs = mfccs.reshape(1, -1)
            # print(mfccs)
            predict_prob = model.predict(mfccs)
            # print(predict_prob)
            predicted_label = np.argmax(predict_prob, axis=1)
            # print(predicted_label)
            for i in predicted_label:
                print(classes[i])
                ans.append(classes[i])
            return str(ans[-1])
    return "files not saved"






if __name__ == '__main__':
    app.run()
