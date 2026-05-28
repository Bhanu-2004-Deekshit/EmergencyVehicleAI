from flask import Flask, render_template, request, redirect, url_for
import tensorflow as tf
import numpy as np
import cv2
import librosa
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/'

# Load Models
visual_model = tf.keras.models.load_model('models/emergency_vehicle_model.h5')
audio_model = tf.keras.models.load_model('models/siren_model.h5')

def process_audio(file_path):
    audio, sr = librosa.load(file_path, duration=3.0, res_type='kaiser_fast')
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    return np.mean(mfccs.T, axis=0).reshape(1, 40)

@app.route('/')
def about():
    return render_template('about.html')

@app.route('/video', methods=['GET', 'POST'])
def video_detector():
    result = None
    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        path = os.path.join(app.config['UPLOAD_FOLDER'], 'current_v.jpg')
        file.save(path)
        img = cv2.imread(path)
        img = cv2.resize(img, (224, 224)) / 255.0
        v_pred = visual_model.predict(np.expand_dims(img, axis=0))[0][0]
        result = "EMERGENCY VEHICLE DETECTED" if v_pred > 0.5 else "NORMAL VEHICLE"
    return render_template('video.html', result=result)

@app.route('/audio', methods=['GET', 'POST'])
def audio_detector():
    result = None
    if request.method == 'POST' and 'audio' in request.files:
        file = request.files['audio']
        path = os.path.join(app.config['UPLOAD_FOLDER'], 'current_a.wav')
        file.save(path)
        features = process_audio(path)
        a_pred = np.argmax(audio_model.predict(features))
        classes = ['Ambulance', 'Police', 'Firetruck', 'Normal Traffic']
        result = classes[a_pred]
    return render_template('audio.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)