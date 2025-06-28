from flask import Flask, render_template, request, redirect, url_for, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import uuid
from datetime import datetime
import csv

app = Flask(__name__)
MODEL_PATH = 'model/skin_cancer_model.h5'
UPLOAD_FOLDER = 'static/uploads'
LOG_CSV = 'prediction_log.csv'

# Ensure folders exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load trained model
model = load_model(MODEL_PATH)


@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    filename = None
    if request.method == 'POST':
        file = request.files['image']
        if file:
            unique_filename = f"{uuid.uuid4().hex}_{file.filename}"
            filepath = os.path.join(UPLOAD_FOLDER, unique_filename)
            file.save(filepath)

            # Preprocess and predict
            img = load_img(filepath, target_size=(224, 224))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)
            pred = model.predict(img_array)[0][0]
            prediction = "Malignant (High Risk Tumor)" if pred > 0.5 else "Benign (Low Risk Tumor)"

            # Log prediction
            with open(LOG_CSV, "a", newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([datetime.now(), unique_filename, prediction, round(float(pred), 4)])

            filename = unique_filename
    return render_template('index.html', prediction=prediction, filename=filename)


@app.route('/history')
def history():
    entries = []
    if os.path.exists(LOG_CSV):
        with open(LOG_CSV, newline='') as csvfile:
            reader = csv.reader(csvfile)
            entries = list(reader)
    return render_template('history.html', entries=entries)


@app.route('/download-log')
def download_log():
    return send_file(LOG_CSV, as_attachment=True)


if __name__ == '__main__':
    import os
    port = int(os.environ.get("PORT", 8080))
    app.run(debug=False, host='0.0.0.0', port=port)

