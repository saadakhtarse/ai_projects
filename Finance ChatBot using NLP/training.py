import random       # For generating random responses
import json         # To parse intent.json file
import pickle       # For data serialization
import numpy as np

import nltk         # Natural Language Toolkit
nltk.download('wordnet')  # Optional: Download WordNet
from nltk.stem import WordNetLemmatizer  # For lemmatizing words

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD

# Initialize lemmatizer
lemmatizer = WordNetLemmatizer()



# Preprocess data -------------------------------------------------------------------------------------------------------------

# Load intents from JSON file
intents = json.loads(open('DataChatBot.json').read())

# Initialize lists and variables
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']  # Characters to ignore when tokenizing words


# Process intents to extract words, classes, and documents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # Tokenize each pattern into words
        word_list = nltk.word_tokenize(pattern)     # Token -> [(I love coding) -> ("I", "love", "coding")]
        words.extend(word_list)                     # Add each list of tokens to the overall set of words
        documents.append((word_list, intent['tag']))    # Each document is a tuple containing a list of tokens and it's
        if intent['tag'] not in classes:               # If the class hasn't been added yet, add it
            classes.append(intent['tag'])               # Add new class to classes list 

# Lemmatize words, remove special characters, remove duplicates, and sort       
words = [lemmatizer.lemmatize(word) for word in words if word not in ignore_letters]    # Remove any special characters and lemmatise
words = sorted(set(words))                 # Remove duplicates and sort alphabetically

classes = sorted(set(classes))              # Sort classes alphabetically

# Save processed words and classes to binary files
pickle.dump(words, open('words.pkl','wb'))           # Save words to a binary file
pickle.dump(classes, open('classes.pkl', 'wb'))      # Save classes to a binary file



# Prepare Training Data -------------------------------------------------------------------------------------------------------------

training = []
output_empty = [0] * len(classes)          # Create an empty vector representing zero probabilities for all classes

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)
    
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Shuffle and convert training data to numpy array
random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])              # Extract patterns from each document (i.e., input)
train_y = list(training[:, 1])              # Extract responses from each document   (i.e., output)



# Build Neural Network -------------------------------------------------------------------------------------------------------------

model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))     # Scalse results in output layer, so they all add up to one. So, we have percentage of how likely it is to have that output

# Use SGD optimizer
sgd = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
model.save('chatbot_model.h5', hist)

print('Model Train Process Done!')




