from preprocess import tokenizacao
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow import keras
import pickle
import json
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping,ModelCheckpoint

df = pd.read_csv("data/dataset.csv")
max_len = 10
x, tokenizer = tokenizacao(df,max_len)

y = df['emocao']
y = pd.get_dummies(y)

emotion_labels = y.columns.tolist()
with open('models/emotion_labels.json', 'w', encoding='utf-8') as f:
    json.dump(emotion_labels, f, ensure_ascii=False, indent=2)

with open('models/tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

x = np.array(x)
y = np.array(y)


X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = keras.Sequential([
    layers.Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=128, mask_zero=True),
    layers.Bidirectional(layers.LSTM(64, return_sequences=True)),
    layers.GlobalMaxPooling1D(),
    layers.Dense(64, activation='relu', kernel_regularizer='l2'),
    layers.Dropout(0.5),
    layers.Dense(5, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
early = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint("models/emocoes_best.keras", save_best_only=True, monitor='val_loss')
model.fit(X_train, y_train, epochs=40, batch_size=16, validation_data=(X_test, y_test),callbacks=[early,checkpoint])

model.save("models/emocoes.keras")

loss, acc = model.evaluate(X_test, y_test)
print(f"Teste - Loss: {loss:.4f}, Accuracy: {acc:.4f}")
