import numpy as np
from flask import Flask, request, jsonify, render_template, flash,url_for,redirect,session

import pandas as pd
import tensorflow as tf
import keras
from keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
app = Flask(__name__)

def init():
	global model
	model = tf.keras.models.load_model('model.h5')
@app.route('/', methods=['GET','POST'])
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST','GET'])
def predict():
		tokenizer = tf.keras.preprocessing.text.Tokenizer()
		data=open('dataset.txt',encoding="utf8").read()
		corpus = data.lower().split("\n")
		tokenizer.fit_on_texts(corpus)
		if request.method=='POST':
			seed_text = request.form['seed_text']
			next_words = request.form['next_words']
			for _ in range(int(next_words)):
				token_list = tokenizer.texts_to_sequences([seed_text])[0]
				token_list = pad_sequences([token_list], maxlen=13, padding='pre')
				predicted = model.predict_classes(token_list, verbose=0)
				output_word = ""
				for word, index in tokenizer.word_index.items():
					if index == predicted:
						output_word = word
						break
				seed_text += " " + output_word
			
			return render_template('index.html',p=seed_text)
		else:
			return render_template('index.html')
			

if (__name__ == "__main__"):
	init()
	app.run(debug=True)
