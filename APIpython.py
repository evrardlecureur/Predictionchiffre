from flask import Flask, request, jsonify, send_file
import numpy as np
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model('DigitRecognizerV2.h5')

@app.route('/')
def index():
    return send_file('./templates/index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    features = data.get('features', None)
    if features is None or len(features) != 28*28:
        return jsonify({'error': 'pas le bon format'}), 400

    # reshape pour le model
    x = np.array(features, dtype='float32')
    x = x.reshape(1, -1)

    probs = model.predict(x)[0]      # shape = (10,)
    pred_class = int(np.argmax(probs))

    maxprob = float(np.max(probs))*100
    # ce que va prendre le js 
    return jsonify({
        'prediction': pred_class,
        'probabilities': maxprob
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
