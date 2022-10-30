from flask import Flask, render_template, request, send_from_directory, Response
from flask_cors import CORS, cross_origin
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims
import numpy as np
import os

app = Flask(__name__)
CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('./model_6.h5')

class_dict = {
    0: 'Bacterial spot', 
    1: 'Early blight', 
    2: 'Late blight', 
    3: 'Leaf Mold', 
    4: 'Septoria leaf spot', 
    5: 'Spider mites Two-spotted spider mite', 
    6: 'Target Spot', 
    7: 'Tomato Yellow Leaf Curl Virus', 
    8: 'Tomato mosaic virus', 
    9: 'healthy'
}

labels = ['Bacterial spot', 'Early blight', 'Late blight', 'Leaf Mold', 'Septoria leaf spot', 'Spider mites Two-spotted spider mite', 'Target Spot', 'Tomato Yellow Leaf Curl Virus', 'Tomato mosaic virus', 'healthy']

def predict_label(img_path):
    loaded_img = load_img(img_path, target_size=(256, 256))
    img_array = img_to_array(loaded_img) / 255.0
    img_array = expand_dims(img_array, 0)
    # predicted_bit = np.round(model.predict(img_array)[0][0]).astype('int')
    x = np.stack([img_array], axis=0)
    y = model.predict(x)
    # print(y)
    # print(np.max(y))
    # predict = labels[np.argmax(y)] + " = "  + str(np.max(y))
    # return predict
    return 'ada'

@app.route('/', methods=['GET', 'POST'])
@cross_origin()
def index():
    if request.method == 'POST':
        if  request.files:
            image = request.files['image']
            img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(img_path)
            prediction = predict_label(img_path)
            return prediction

    return 'kosong'

@app.route('/display/<filename>')
@cross_origin()
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)