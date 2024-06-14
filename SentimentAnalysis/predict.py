import keras
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
import openpyxl
import csv

model = keras.models.load_model('sentiment_analysis_model.keras')
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

df = pd.read_csv('predict-1.csv',encoding='latin-1 ')

print(df['content'])



text_sequence = tokenizer.texts_to_sequences(df['content'])
text_sequence = pad_sequences(text_sequence, maxlen=100)

predicted_rating = model.predict(text_sequence)

ans=[]
p=np.argmax(predicted_rating, axis=1)

for i in range(len(p)):
    if p[i]==0:
        ans.append('-1')
    if p[i]==1:
        ans.append('0')
    if p[i]==2:
        ans.append('1')

filepath = "result.csv"

print(len(ans))

ans2=[]

for i in range(len(ans)):
    temp=[]
    temp.append(ans[i])
    ans2.append(temp)

with open(filepath, 'w', newline='') as f:
    # using csv.writer method from CSV package
    write = csv.writer(f)
    write.writerows(np.array(ans2))


#wb = openpyxl.Workbook()
#ws = wb.active

#for a in ans:
#    ws.append([a])

#wb.save(filepath)
#wb.close()
#print(df['content'])
