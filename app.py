from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__)

# Upload folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# -------------------------------
# Load Trained Model
# -------------------------------
MODEL_PATH = "model/butterfly_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------
# Class Labels (UPDATE if needed)
# -------------------------------
class_labels = sorted(os.listdir("dataset_ready/train"))

# -------------------------------
# Prediction Function
# -------------------------------
def predict_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)

    return class_labels[class_index]

# -------------------------------
# Routes
# -------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    image_path = None

    if request.method == "POST":
        file = request.files["file"]

        if file:
            image_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(image_path)

            prediction = predict_image(image_path)

    return render_template(
        "index.html",
        prediction=prediction,
        image_path=image_path
    )

# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
