import pandas as pd
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

df = pd.read_csv("data/dataset.csv")

def limpaTexto(df):
    df['frase'] = df['frase'].str.lower()
    df['frase'] = df['frase'].str.replace('[^\w\s]','')
    return df

def tokenizacao(df, max_len=10):
    limpaTexto(df)
    frases = df['frase'].astype(str).tolist()

    tokenizer = Tokenizer(oov_token="<OOV>")
    tokenizer.fit_on_texts(frases)

    sequencias = tokenizer.texts_to_sequences(frases)
    padded = pad_sequences(sequencias, maxlen=max_len, padding='post', truncating='post')

    return padded, tokenizer



if __name__ == "__main__":
    df = pd.read_csv("data/dataset.csv")
    df = limpaTexto(df)
    p, tokenizer = tokenizacao(df)
    
    print("Vocabulário criado:", tokenizer.word_index)
    print("Sequências tokenizadas e com padding:")
    print(p)



