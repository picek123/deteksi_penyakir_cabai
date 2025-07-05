import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras.models import model_from_json
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import numpy as np
import PIL
from PIL import Image

app = Flask(__name__)

# Load model
MODEL_ARCHITECTURE = 'model/model_plant_disease_MobileNetV250fix.json'
MODEL_WEIGHTS = 'model/model_plant_disease_weightMobileNetV250fix.weights.h5'

assert os.path.exists(MODEL_ARCHITECTURE), "Model architecture file not found!"
assert os.path.exists(MODEL_WEIGHTS), "Model weights file not found!"

json_file = open(MODEL_ARCHITECTURE, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.load_weights(MODEL_WEIGHTS)
print('@@ Model loaded. Check http://192.168.1.46:5000/templates/')


def model_predict(img_path, model):
    try:
        test_image = load_img(img_path, target_size=(224, 224))  # load image
        print("@@ Got Image for prediction")
    
        test_image = img_to_array(test_image) / 255.0  # normalisasi
        test_image = np.expand_dims(test_image, axis=0)
    
        result = model.predict(test_image)
        confidence = np.max(result)
        pred_index = np.argmax(result, axis=1)[0]  # ambil index prediksi

        # Mapping label
        label_map = {0: 'bercak_daun', 1: 'daun_keriting', 2: 'sehat'}
        label = label_map.get(pred_index, 'unknown')

        print(f"Predicted: {label} with confidence: {confidence:.2f}")

        # Cek confidence < 0.7
        if confidence < 0.8:
            return f"Objek tidak dikenali sebagai daun cabai ", 'error.html'

        if label == 'bercak_daun':
            return "Tanamanmu terkena penyakit bercak daun", 'classification/cacar.html'
        elif label == 'daun_keriting':
           return "Tanamanmu terkena penyakit daun keriting", 'classification/keriting.html'
        elif label == 'sehat':
         return "Tanamanmu sehat", 'classification/sehat.html'

        else:
            return f"Objek tidak dikenali sebagai daun cabai ", 'error.html'

    except Exception as e:
        print(f"Error during prediction: {e}")
        return "Terjadi kesalahan saat prediksi", "error.html"


@app.route("/templates/classification/", methods=['GET', 'POST'])
def classification():
    return render_template('classification/index.html')

@app.route("/templates/tutorial/", methods=['GET', 'POST'])
def tutorial():
    return render_template('tutorial/index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        file = request.files['image']  # get input
        filename = secure_filename(file.filename)
        print("@@ Input posted = ", filename)
        
        file_path = os.path.join('static/user uploaded', filename)
        file.save(file_path)

        print("@@ Predicting class......")
        pred, output_page = model_predict(file_path, model)
        
        return render_template(output_page, pred_output=pred, user_image=file_path)
    return redirect(url_for('home'))

@app.route('/templates/')
def home():
    return render_template('index.html')


if __name__ == "__main__":
    app.debug = True  # Enable debug mode (matikan saat produksi)
    app.run(host='0.0.0.0', port=5000, threaded=True)
