import pandas as pd
import numpy as np

df = pd.read_csv('./data/text_emotion.csv')  


emotions = df['sentiment'].tolist()

unique_labels = list(set(emotions))

label_map = []

for label in unique_labels:
    output_values = np.zeros(len(unique_labels), dtype=np.int)
    output_values [unique_labels.index(label)] = 1
    label_map.append({'name': label , 'value': output_values })


def get_one_hot_encoded_array_for_label(label):
    for existing in label_map:
        if existing['name'] == label:
            return np.array(existing['value'])

labels = []
for emotion in emotions:
    labels.append(get_one_hot_encoded_array_for_label(emotion))

print(labels[:10])

labels = np.array(labels)

content = df['content'].tolist()

vocab_size = 10000
embedding_dim = 64
max_length = 120
turnc_type = 'post'
oov_token = "<00V>"

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

tokenizer = Tokenizer(num_words=vocab_size,oov_token=oov_token)
tokenizer.fit_on_texts(content)

word_index = tokenizer.word_index

sequences = tokenizer.texts_to_sequences(content)
padded = pad_sequences(sequences, padding=turnc_type,truncating=turnc_type,maxlen=max_length)

# print(padded[0])
# print(padded.shape)



model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,input_length=max_length),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(13,activation='softmax')
])



model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

NUM_EPOCHS = 100
model.fit(padded, labels,epochs=NUM_EPOCHS)
