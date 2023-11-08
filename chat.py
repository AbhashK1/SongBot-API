import json
import random
import nltk
import numpy as num
import pickle
import pandas as pd
import tensorflow as tensorF
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Dropout

nltk.download("punkt")
nltk.download("wordnet")

loaded_model = None  

# Lazy loading of the model
def load_model():
    global loaded_model
    if loaded_model is None:
        loaded_model = tensorF.keras.models.load_model('model.h5')


def load_data():
    with open('data.json', 'r') as file:
        ourData = json.load(file)

    with open('newWords.pkl', 'rb') as file:
        newWords = pickle.load(file)

    with open('ourClasses.pkl', 'rb') as file:
        ourClasses = pickle.load(file)

    tracks1 = pd.read_csv('SpotifyFeatures.csv')

    return ourData, newWords, ourClasses, tracks1

lm = None

def get_lemmatizer():
    global lm
    if lm is None:
        lm = nltk.stem.WordNetLemmatizer()
    return lm

def ourText(text):
    lemmatizer = get_lemmatizer()
    newtkns = nltk.word_tokenize(text)
    newtkns = [lemmatizer.lemmatize(word) for word in newtkns]
    return newtkns


def wordBag(text, vocab):
    newtkns = ourText(text)
    bagOwords = [0] * len(vocab)
    for w in newtkns:
        for idx, word in enumerate(vocab):
            if word == w:
                bagOwords[idx] = 1
    return num.array(bagOwords)

def pred_class(text, vocab, labels):
    load_model()  # Load the model when needed
    bagOwords = wordBag(text, vocab)
    ourResult = loaded_model.predict(num.array([bagOwords]))[0]
    newThresh = 0.2
    yp = [[idx, res] for idx, res in enumerate(ourResult) if res > newThresh]

    yp.sort(key=lambda x: x[1], reverse=True)
    newList = []
    for r in yp:
        newList.append(labels[r[0])
    return newList

def getRes(firstlist, fJson):
    tag = firstlist[0]
    listOfIntents = fJson["intents"]
    genre_list = ['Movie', 'R&B', 'A Capella', 'Alternative', 'Country', 'Dance', 'Electronic',
                  'Anime', 'Folk', 'Blues', 'Opera', 'Hip-Hop', "Children's Music", "Children’s Music",
                  'Rap', 'Indie', 'Classical', 'Pop', 'Reggae', 'Reggaeton', 'Jazz', 'Rock', 'Ska',
                  'Comedy', 'Soul', 'Soundtrack', 'World']

    if tag in genre_list:
        genre_data = tracks1[tracks1['genre'] == tag]
        random_song = genre_data.sample(n=1)
        selected_artist = random_song['artist_name'].values[0]
        selected_track = random_song['track_name'].values[0]
        recommendations = [
          f"How about '{selected_track}' by {selected_artist}",
          f"Consider listening to the '{selected_track}' by {selected_artist} theme.",
          f"You might enjoy '{selected_track}' by {selected_artist} soundtrack.",
          f"Check out the '{selected_track}' theme by {selected_artist}.",
          f"Explore the '{selected_track}' soundtrack by {selected_artist}."
        ]
        ourResult = random.choice(recommendations)
    else:
        for i in listOfIntents:
            if i["tag"] == tag:
                ourResult = random.choice(i["responses"])
                break
    return ourResult

def chatRes(data):
    ourData, newWords, ourClasses, tracks1 = load_data() 
    newMessage = data.lower()
    intents = pred_class(newMessage, newWords, ourClasses)
    ourResult = getRes(intents, ourData)
    return ourResult
