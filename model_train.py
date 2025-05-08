import json
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalAveragePooling1D
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping

# Veriyi yükle
with open("intents.json") as file:
    data = json.load(file)

# Eğitim verilerini hazırla
training_sentences = []
training_labels = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        training_sentences.append(pattern)
        training_labels.append(intent['tag'])

    if intent['tag'] not in labels:
        labels.append(intent['tag'])

number_of_classes = len(labels)

# Label encoding
label_encoder = LabelEncoder()
training_labels = label_encoder.fit_transform(training_labels)

# Tokenizer
vocab_size = 1000
max_len = 20
ovv_token = "<OOV>"
embedding_dim = 16

tokenizer = Tokenizer(num_words=vocab_size, oov_token=ovv_token)
tokenizer.fit_on_texts(training_sentences)
sequences = tokenizer.texts_to_sequences(training_sentences)
padded_sequences = pad_sequences(sequences, truncating='post', maxlen=max_len)

# Modeli oluştur
model = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=max_len),
    GlobalAveragePooling1D(),
    Dense(16, activation="relu"),
    Dense(16, activation="relu"),
    Dense(number_of_classes, activation="softmax")
])

model.compile(loss='sparse_categorical_crossentropy', optimizer="adam", metrics=["accuracy"])
model.summary()

# Early stopping ile eğit
early_stop = EarlyStopping(monitor='loss', patience=10, restore_best_weights=True)
history = model.fit(padded_sequences, np.array(training_labels), epochs=200, callbacks=[early_stop])

# Modeli kaydet
model.save("chat_model.h5")
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)
with open("label_encoder.pkl", "wb") as encoder_file:
    pickle.dump(label_encoder, encoder_file)
