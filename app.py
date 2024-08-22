from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from keras.models import load_model
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import json
import random

# Initialize the Flask application
app = Flask(__name__)

# Load the model, intents, and other necessary data
model = tf.keras.models.load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
lemmatizer = WordNetLemmatizer()

# Define functions for text preprocessing and prediction
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
                if show_details:
                    print(f"found in bag: {w}")
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(text):
    ints = predict_class(text, model)
    res = get_response(ints, intents)
    return res

# Define the route for the homepage
@app.route("/")
def index():
    return render_template('chat.html')

# Define the route for getting a response from the bot
@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    response = chatbot_response(msg)
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
