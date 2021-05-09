from flask import Flask, jsonify, request, render_template
from base64 import b64decode, b64encode
from predicting_output import prediction
import os
app = Flask(__name__)


@app.route('/')
def home():
    return 'This is Home Page!'


@app.route('/predict', methods=['POST'])
def predict():
    if(request.method == 'POST'):
        b64string = request.form['base64data']
        decoded = b64decode(b64string)
        with open("temp.tif", 'wb') as file:
            file.write(decoded)
        #playsound('temp.tif')
        return jsonify({
            "The prediction is:": prediction('temp.tif'),
        })

if __name__ == "__main__":
    app.run(port=3333, debug=True)
