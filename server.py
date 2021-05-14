# Import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.feature_extraction.text import CountVectorizer

# Load the model
model = pickle.load(open('model/NB_spam_model.pkl','rb'))
vectorizer = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open('model/feature.pkl','rb')))

app = Flask(__name__)

@app.route('/isAlive')
def index():
    return "true"

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/api',methods=['POST'])
def api_predict():
    message = request.form['message']
    data = [message]
    vect = vectorizer.transform(data).toarray()
    prediction = model.predict(vect)
    print(prediction)
    # Take the first value of prediction
    output = prediction[0]
    print(output)
    return jsonify(int(output))

@app.route('/app',methods=['POST'])
def predict():
    # Get the data from the POST request.
    # Make prediction using model loaded from disk as per the data.
    message = request.form['message']
    data = [message]
    vect = vectorizer.transform(data).toarray()
    prediction = model.predict(vect)
    print(prediction)
    # Take the first value of prediction
    output = prediction[0]
    print(output)
    #return jsonify(int(output))
    return render_template('result.html', prediction = prediction)


if __name__ == '__main__':
    app.run(port=8111, debug=True)
