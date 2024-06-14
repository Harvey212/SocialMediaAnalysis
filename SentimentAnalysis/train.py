import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
import pickle


df = pd.read_csv('data-1.csv',encoding='latin-1 ')

df = df.sample(frac=1).reset_index(drop=True)

tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(df['content'])
word_index = tokenizer.word_index
sequences = tokenizer.texts_to_sequences(df['content'])
padded_sequences = pad_sequences(sequences, maxlen=100, truncating='post')


sentiment_labels = pd.get_dummies(df['category']).values

#print(sentiment_labels)
x_train, x_test, y_train, y_test = train_test_split(padded_sequences, sentiment_labels, test_size=0.2)

model = Sequential()
model.add(Embedding(5000, 100, input_length=100))
model.add(Conv1D(64, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3, activation='softmax'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

model.fit(padded_sequences, sentiment_labels, epochs=10, batch_size=32, validation_data=(x_test, y_test))
model.save('sentiment_analysis_model.keras')

with open('tokenizer.pickle', 'wb') as handle:
	pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)