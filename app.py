import os
import io # Import io for in-memory file handling
import base64 # Import base64 for image display
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer # This import seems to be for a production server setup, keeping it.
from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions # Not used in the provided code, but keeping for completeness.
from keras.models import load_model, model_from_json # Keeping load_model though model_from_json is used.
from keras.preprocessing import image # Not directly used for loading, but img_to_array is from here.
from keras.preprocessing.image import load_img, img_to_array # img_to_array is used. load_img will be replaced.
import numpy as np
from PIL import Image # Import Image from PIL for image manipulation

# Set TensorFlow environment variable
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

app = Flask(__name__)

# Load model architecture and weights
MODEL_ARCHITECTURE = 'model/model_plant_disease_MobileNetV250fix.json'
MODEL_WEIGHTS = 'model/model_plant_disease_weightMobileNetV250fix.weights.h5'

# Assert that model files exist before attempting to load
assert os.path.exists(MODEL_ARCHITECTURE), f"Model architecture file not found at: {MODEL_ARCHITECTURE}"
assert os.path.exists(MODEL_WEIGHTS), f"Model weights file not found at: {MODEL_WEIGHTS}"

# Load model from JSON and then load weights
try:
    with open(MODEL_ARCHITECTURE, 'r') as json_file:
        loaded_model_json = json_file.read()
    model = model_from_json(loaded_model_json)
    model.load_weights(MODEL_WEIGHTS)
    print('@@ Model loaded successfully. Check http://192.168.1.46:5000/templates/')
except Exception as e:
    print(f"Error loading model: {e}")
    # Exit or handle error appropriately if model fails to load
    exit(1)


def model_predict(image_file_object, model):
    """
    Predicts the class of the plant leaf image.

    Args:
        image_file_object: An in-memory file-like object (e.g., io.BytesIO)
                           containing the image data.
        model: The loaded Keras model for prediction.

    Returns:
        A tuple containing:
        - A string message about the prediction result.
        - The path to the HTML template to render.
    """
    try:
        # Open the image directly from the in-memory object using PIL
        test_image = Image.open(image_file_object)
        # Resize the image to the target size expected by the model (224x224)
        # Using Image.LANCZOS for high-quality downsampling
        test_image = test_image.resize((224, 224), Image.LANCZOS)
        print("@@ Image loaded from memory for prediction.")

        # Convert the PIL Image to a NumPy array and normalize pixel values to [0, 1]
        test_image = img_to_array(test_image) / 255.0
        # Add an extra dimension to fit the model's expected input shape (batch_size, height, width, channels)
        test_image = np.expand_dims(test_image, axis=0)

        # Perform prediction
        result = model.predict(test_image)
        confidence = np.max(result) # Get the maximum confidence score
        pred_index = np.argmax(result, axis=1)[0] # Get the index of the class with highest confidence

        # Define the mapping from prediction index to human-readable label
        label_map = {0: 'bercak_daun', 1: 'daun_keriting', 2: 'sehat'}
        label = label_map.get(pred_index, 'unknown') # Get the label, default to 'unknown'

        print(f"Predicted: {label} with confidence: {confidence:.2f}")

        # Check confidence threshold (0.8 as per user's implicit request)
        if confidence < 0.8:
            return "Objek tidak dikenali sebagai daun cabai", 'error.html'

        # Return appropriate message and template based on the predicted label
        if label == 'bercak_daun':
            return "Tanamanmu terkena penyakit bercak daun", 'classification/cacar.html'
        elif label == 'daun_keriting':
           return "Tanamanmu terkena penyakit daun keriting", 'classification/keriting.html'
        elif label == 'sehat':
            return "Tanamanmu sehat", 'classification/sehat.html'
        else:
            # Fallback for 'unknown' label or other unexpected cases
            return "Objek tidak dikenali sebagai daun cabai", 'error.html'

    except Exception as e:
        # Log the error and return a generic error message and template
        print(f"Error during prediction: {e}")
        return "Terjadi kesalahan saat prediksi", "error.html"


@app.route("/templates/classification/", methods=['GET', 'POST'])
def classification():
    """Renders the classification page."""
    return render_template('classification/index.html')

@app.route("/templates/tutorial/", methods=['GET', 'POST'])
def tutorial():
    """Renders the tutorial page."""
    return render_template('tutorial/index.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    """
    Handles image upload and initiates the prediction process.
    The uploaded image is processed in memory.
    """
    if request.method == 'POST':
        # Check if 'image' file is present in the request
        if 'image' not in request.files:
            return render_template('error.html', pred_output="No image part in the request.")
        
        file = request.files['image'] # Get the uploaded file object

        # If the user does not select a file, the browser submits an
        # empty file without a filename.
        if file.filename == '':
            return render_template('error.html', pred_output="No selected file.")

        if file:
            # Read the image data into an in-memory BytesIO object
            img_bytes = file.read()
            img_io = io.BytesIO(img_bytes)
            
            print("@@ Predicting class......")
            # Call model_predict with the in-memory image object
            pred, output_page = model_predict(img_io, model)
            
            # For displaying the image on the result page, convert to base64 data URL
            encoded_image = base64.b64encode(img_bytes).decode('utf-8')
            user_image_data_url = f"data:{file.mimetype};base64,{encoded_image}"

            return render_template(output_page, pred_output=pred, user_image=user_image_data_url)
    
    # Redirect to home if accessed via GET or if no file was uploaded in POST
    return redirect(url_for('home'))

@app.route('/templates/')
def home():
    """Renders the home page."""
    return render_template('index.html')

@app.route('/')
def root_redirect():
    """Redirects the root URL to /templates/."""
    return redirect(url_for('home'))


if __name__ == "__main__":
    # Enable debug mode for development (set to False for production)
    app.debug = True 
    # Run the Flask application
    # host='0.0.0.0' makes the server accessible from any IP address
    # threaded=True allows handling multiple requests concurrently
    app.run(host='0.0.0.0', port=5000, threaded=True)

