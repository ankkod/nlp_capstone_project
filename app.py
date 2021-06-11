# Import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
from flask_cors import cross_origin
import tensorflow as tf
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
import fasttext

# Load the model
#model = pickle.load(open('model/NB_spam_model.pkl','rb'))
#vectorizer = CountVectorizer(decode_error="replace", vocabulary=pickle.load(open('model/feature.pkl','rb')))
# loading
#with open('model/tokenizer.pl', 'rb') as handle:
tokenizer = pickle.load(open('model/tokenizer.pl','rb'))
bidir_model = tf.keras.models.load_model('model/bidir-lstm.h5')
lstm_model = tf.keras.models.load_model('model/lstm_model_all.h5')
fasttext_model = fasttext.load_model("model/fasttext_model.ftz")
groups = pd.read_csv('model/final_data.csv')

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
    description = [message]
    model = data['model']
    actual_label = ''
    if model == 'Bi-Directional_LSTM' or model == 'LSTM':
        # vect = vectorizer.transform(data).toarray()
        test_description = tokenizer.texts_to_sequences(description)
        test_seq = pad_sequences(test_description)
        print(test_seq)
        #print(model.predict(test_seq[0])[0].argmax())
        if model == 'Bi-Directional_LSTM':
            prediction = bidir_model.predict(test_seq[0])[0].argmax()
        else:
            prediction = lstm_model.predict(test_seq[0])[0].argmax()
        actual_label = check_group(prediction)
        print(actual_label)
        # Take the first value of prediction
        #output = prediction[0]
        #print(output)
    else:
        res = fasttext_model.predict(message)
        actual_label = res[0][0].replace('__label__','')
    return jsonify(actual_label)


@app.route('/app',methods=['POST'])
def predict():
    # Get the data from the POST request.
    # Make prediction using model loaded from disk as per the data.
    message = request.form['message']
    res = fasttext_model.predict(message)
    actual_label = res[0][0].replace('__label__','')
    # data = [message]
    # test_description = tokenizer.texts_to_sequences(data)
    # test_seq = pad_sequences(test_description)
    # prediction = lstm_model.predict(test_seq[0])[0].argmax()
    # print(prediction)
    # actual_label = check_group(prediction)
    # Take the first value of prediction
    #output = prediction[0]
    #print(output)
    #return jsonify(int(output))
    return render_template('result.html', prediction = actual_label)

def check_group(label):
    return groups.loc[groups['Grp_New']==label]['Assignment_group'].iloc[0]


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(debug=True, port=5000)
