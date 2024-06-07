from flask import Flask, Response, request, url_for, jsonify
from flask_cors import CORS
from pickle import load

app = Flask(__name__)
CORS(app)

@app.route("/iris", methods=['POST'])
def iris():
    
    with open('models/iris.pkl', 'rb') as f:
        model = load(f)

    labels = [
        {'name': 'Setosa', 'image': url_for('static', filename='setosa.jpg', _external=True) },
        {'name': 'Versicolor', 'image': url_for('static', filename='versicolor.jpg', _external=True) },
        {'name': 'Virginica', 'image': url_for('static', filename='virginica.jpg', _external=True) }
    ]
    
    data = request.get_json()
    
    petalLength = float(data.get('petalLength'))
    petalWidth = float(data.get('petalWidth'))
    sepalLength = float(data.get('sepalLength'))
    sepalWidth = float(data.get('sepalWidth'))

    prediction = model.predict([[petalLength, petalWidth, sepalLength, sepalWidth]])[0]
    print(type(prediction))
    prediction_label = labels[prediction];
    
    response = jsonify(prediction_label)
    return response

@app.route("/digits")
def digits():
    return "Hand written digits classifier demonstration"
