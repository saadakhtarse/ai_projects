import random       # For generating random responses
import json         # To parse intent.json file
import pickle       # For data serialization
import numpy as np

# Import Speak library for text-to-speech functionality
from speak import Speak

import tensorflow as tf
from keras.models import load_model

import nltk         # Natural Language Tool Kit
# nltk.download('wordnet')   # Download WordNet (optional)
from nltk.stem import WordNetLemmatizer # Reduce (lemmatizing) word to its stem (Words: work, working, worked -> work)

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()



# Load  the pre-trained model and tokeniser -----------------------------------------------------------------------------

# Load intents, words, classes, and pre-trained model
intents = json.loads(open('DataChatBot.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')



# Functions -------------------------------------------------------------------------------------------------------------

def clean_up_sentence(sentence):
    # Clean up sentences by tokenizing and lemmatizing words
    sentence_words = nltk.word_tokenize(sentence) 
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# Bag of Words (BOW): Convert snetences into bag of words, into a list full of zeros and ones, indicate that word is there or not.
def bag_of_words(sentence):
    # Convert sentences into bag of words
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# Predict the intent class based on the input sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)                    # Create Bag of Words(BOW)
    result_ = model.predict(np.array([bow]))[0]     # Predict result based on Bag of Words(BOW)

    # Set threshold
    ERROR_THRESHOLD=0.25        # We had 'softmax' function, which means, each position in output is going to likelihood of that class being the result
    # and if it be below 25%. We dont take that in
    results = [[i, r] for i, r in enumerate(result_) if r > ERROR_THRESHOLD]

    # Sort the results by Probability
    results.sort(key=lambda x: x[1], reverse=True)        # Take 1st index everytime, sort it in reverse order(descending order)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})     # Actual result, the first one in the index
    return return_list

def get_response(intents_list, intents_json):
    # Get a response based on the predicted intent
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

print("Bot is running...")



# ChatBot Starts here -------------------------------------------------------------------------------------------------------------
flag = True
while (flag):
    message = input("User: ")
    if message != 'quit':
        ints = predict_class(message)
        result = get_response(ints, intents)
        Speak(result)
        # print(result)
    else:
        flag = False
        Speak("Good Bye!")
        # print("Good Bye!")






