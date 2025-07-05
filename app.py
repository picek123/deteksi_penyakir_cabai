import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import io # Import the io module to handle in-memory file operations

app = Flask(__name__)

# Load model
MODEL_ARCHITECTURE = 'model/model_plant_disease_MobileNetV250fix.json'
MODEL_WEIGHTS = 'model/model_plant_disease_weightMobileNetV250fix.weights.h5'

# Ensure model files exist before attempting to load
assert os.path.exists(MODEL_ARCHITECTURE), "Model architecture file not found!"
assert os.path.exists(MODEL_WEIGHTS), "Model weights file not found!"

# Load the model architecture from JSON file
json_file = open(MODEL_ARCHITECTURE, 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

# Load the model weights
model.load_weights(MODEL_WEIGHTS)
print('@@ Model loaded. Check http://192.168.1.46:5000/templates/')


def model_predict(img_file_stream, model):
    """
    Performs prediction on an image stream using the loaded Keras model.

    Args:
        img_file_stream: A file-like object (e.g., Flask's FileStorage object)
                         containing the image data.
        model: The loaded Keras model for prediction.

    Returns:
        A tuple containing the prediction message and the template page to render.
    """
    try:
        # Open the image directly from the file stream using PIL (Pillow)
        # io.BytesIO(img_file_stream.read()) creates an in-memory binary stream
        # from the uploaded file's content.
        img = Image.open(io.BytesIO(img_file_stream.read()))

        # Resize the PIL image to the target size expected by the model
        test_image = img.resize((224, 224))
        print("@@ Got Image for prediction")

        # Convert the PIL image to a NumPy array and normalize pixel values
        test_image = img_to_array(test_image) / 255.0
        # Add a batch dimension to the image array
        test_image = np.expand_dims(test_image, axis=0)

        # Make the prediction
        result = model.predict(test_image)
        confidence = np.max(result) # Get the highest confidence score
        pred_index = np.argmax(result, axis=1)[0] # Get the index of the highest prediction

        # Define the mapping from prediction index to human-readable label
        label_map = {0: 'bercak_daun', 1: 'daun_keriting', 2: 'sehat'}
        label = label_map.get(pred_index, 'unknown') # Get the label, default to 'unknown'

        print(f"Predicted: {label} with confidence: {confidence:.2f}")

        # Check if the confidence is below a certain threshold
        if confidence < 0.8:
            return "Objek tidak dikenali sebagai daun cabai ", 'error.html'

        # Return appropriate message and template based on the predicted label
        if label == 'bercak_daun':
            return "Tanamanmu terkena penyakit bercak daun", 'classification/cacar.html'
        elif label == 'daun_keriting':
            return "Tanamanmu terkena penyakit daun keriting", 'classification/keriting.html'
        elif label == 'sehat':
            return "Tanamanmu sehat", 'classification/sehat.html'
        else:
            return "Objek tidak dikenali sebagai daun cabai ", 'error.html'

    except Exception as e:
        # Log any errors that occur during prediction
        print(f"Error during prediction: {e}")
        return "Terjadi kesalahan saat prediksi", "error.html"


@app.route("/templates/classification/", methods=['GET', 'POST'])
def classification():
    """Renders the classification index page."""
    return render_template('classification/index.html')

@app.route("/templates/tutorial/", methods=['GET', 'POST'])
def tutorial():
    """Renders the tutorial index page."""
    return render_template('tutorial/index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    """
    Handles image uploads and triggers the prediction process.
    The image is processed directly from the request without saving to disk.
    """
    if request.method == 'POST':
        # Get the uploaded file from the request
        file = request.files['image']
        print("@@ Input posted (image will be processed in-memory)")

        # Pass the FileStorage object directly to model_predict.
        # model_predict will handle reading the stream.
        pred, output_page = model_predict(file, model)

        # Render the appropriate template with the prediction output.
        # user_image is no longer passed as the image is not saved to a file path.
        return render_template(output_page, pred_output=pred)
    # If it's a GET request, redirect to the home page
    return redirect(url_for('home'))

@app.route('/templates/')
def home():
    """Renders the home page."""
    return render_template('index.html')


if __name__ == "__main__":
    app.debug = True  # Enable debug mode (set to False for production)
    app.run(host='0.0.0.0', port=5000, threaded=True)
