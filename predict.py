import numpy as np
from tensorflow import keras
from preprocess import limpaTexto
from utils import get_emoji
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
import json

model = keras.models.load_model("models/emocoes_best.keras")
max_len = 10

with open('models/tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
with open('models/emotion_labels.json', 'r', encoding='utf-8') as f:
    emotion_labels = json.load(f)

def predict_emoji(text):
    df_temp = pd.DataFrame({'frase': [text]})
    df_temp = limpaTexto(df_temp)
    sequences = tokenizer.texts_to_sequences(df_temp['frase'])
    padded = pad_sequences(sequences, maxlen=max_len, padding='post', truncating='post')
    prediction = model.predict(padded)
    predicted_label = emotion_labels[np.argmax(prediction)]
    return get_emoji(predicted_label)

if __name__ == "__main__":
    while True:
        user_input = input("\nDigite uma frase (ou 'sair' para terminar): ")
        if user_input.lower() == 'sair':
            break
        emoji = predict_emoji(user_input)
        print(f"Emoji: {emoji}")