from flask import Flask, render_template, request, send_from_directory, Response
from flask_cors import CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow import expand_dims
import numpy as np
import os

app = Flask(__name__)
CORS(app)
app.config['UPLOAD_FOLDER'] = './static/uploads/'
model = load_model('model_6.h5')

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

def predict_label(img_path):
    loaded_img = load_img(img_path, target_size=(256, 256))
    img_array = img_to_array(loaded_img) / 255.0
    img_array = expand_dims(img_array, 0)
    predicted_bit = np.round(model.predict(img_array)[0][0]).astype('int')
    return class_dict[predicted_bit]

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if request.files:
            # image = request.files['file']
            # img_path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            # image.save(img_path)
            # prediction = predict_label(img_path)
            # return render_template('index.html', uploaded_image=image.filename, prediction=prediction)
            return request.files

    return 'kosong'

@app.route('/display/<filename>')
def send_uploaded_image(filename=''):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)