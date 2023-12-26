from flask import Flask, render_template, request
from PIL import Image
import numpy as np
from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.models import Sequential
import cv2

app = Flask(__name__, template_folder='templates')

model = load_model('mnistCNN.h5')


@app.route('/')
def upload_file():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload_image_file():
    if request.method == 'POST':
        graph = tf.compat.v1.get_default_graph()
        img = Image.open(request.files['file'].stream).convert("L")
        img = img.resize((28, 28))
        im2arr = np.array(img)
        im2arr = im2arr.reshape(1, 28, 28, 1)
        y_pred = model.predict_classes(im2arr)
        predict = str(y_pred[0])
        # 'Predicted Number: ' + str(y_pred[0])
        return render_template("Predict.html", predict=predict)


if __name__ == '__main__':
    print(("* Loading Keras model and Flask starting server..."
           "please wait until server has fully started"))
    app.run(debug=True)
