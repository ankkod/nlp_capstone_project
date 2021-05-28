# Import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
#from sklearn.feature_extraction.text import CountVectorizer
from flask_cors import cross_origin
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model

# Load the model
#model = pickle.load(open('model/NB_spam_model.pkl','rb'))
#vectorizer = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open('model/feature.pkl','rb')))
# loading
#with open('model/tokenizer.pl', 'rb') as handle:
tokenizer = pickle.load(open('model/tokenizer.pl','rb'))

model = tf.keras.models.load_model('model/bidir-lstm_model.h5')

app = Flask(__name__)

@app.route('/isAlive')
@cross_origin()
def index():
    return "true"

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/api',methods=['POST'])
@cross_origin()
def api_predict():
    data = request.get_json(force=True)
    message = data['message']
    data = [message]
    # vect = vectorizer.transform(data).toarray()
    test_description = tokenizer.texts_to_sequences(data)
    test_seq = pad_sequences(test_description)
    print(test_seq)
    print(model.predict(test_seq[0])[0].argmax())
    prediction = model.predict(test_seq[0])[0].argmax()
    print(prediction)
    # Take the first value of prediction
    #output = prediction[0]
    #print(output)
    return jsonify(str(prediction))


@app.route('/app',methods=['POST'])
def predict():
    # Get the data from the POST request.
    # Make prediction using model loaded from disk as per the data.
    message = request.form['message']
    data = [message]
    test_description = tokenizer.texts_to_sequences(data)
    test_seq = pad_sequences(test_description)
    prediction = model.predict(test_seq[0])[0].argmax()
    print(prediction)
    # Take the first value of prediction
    #output = prediction[0]
    #print(output)
    #return jsonify(int(output))
    return render_template('result.html', prediction = prediction)

if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(debug=True, port=5000)
