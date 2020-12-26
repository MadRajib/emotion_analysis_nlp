import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import extract
import utils
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

class EmotionModel():
    def __init__(self,vocab_size,embedding_dim,max_length,turnc_type,oov_token,pad_type):
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.max_length  = max_length
        self.turnc_type = turnc_type
        self.oov_token = oov_token
        self.pad_type = pad_type

    def model(self):
        self.model = tf.keras.Sequential([
            tf.keras.layers.Embedding(self.vocab_size, self.embedding_dim,input_length=self.max_length),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64,return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(13,activation='softmax')
        ])
        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.summary()

    def train(self,padded,labels,test_dataset,num_epochs=5):
        history = self.model.fit(
            padded, 
            labels,
            epochs=num_epochs,
            validation_data=test_dataset
        )
        return history
        
    def eval(self):
        pass
    def test(self,test_x,test_y):
        result = self.model.evaluate(test_x,test_y,verbose=1)
        return result

    def init_tokenizer(self,content):
        self.tokenizer = Tokenizer(num_words=self.vocab_size,oov_token=self.oov_token)
        self.tokenizer.fit_on_texts(content)
    
    def tokenize_sequences(self,content):
        sequences = self.tokenizer.texts_to_sequences(content)
        padded = pad_sequences(sequences, padding=self.pad_type,truncating=self.turnc_type,maxlen=self.max_length)

        return padded

    def plot_graphs(self, history,string):
        plt.plot(history.history[string])
        plt.plot(history.history['val_'+string])
        plt.xlabel("Epochs")
        plt.ylabel(string)
        plt.legend([string, 'val_'+string])
        plt.show()
        



if __name__ == "__main__":

    hyper_param = extract.extract_hyper_param("./model/hyper_param.json")
    model = EmotionModel(**hyper_param)

    # ----------------------------------------------------------------------------------
    # label_dict = extract.extract_mapping("./data/mapping.txt")
    # label_map  = utils.get_label_map(label_dict)

    
    # train_x ,y = extract.extract_content_labels("./data/train")

    
    # model.init_tokenizer(train_x)

    # train_x = model.tokenize_sequences(train_x)

    # train_y = []
    # for emotion in y:
    #     train_y.append(utils.get_one_hot_encoded_array_for_label(emotion,label_dict,label_map))

    # train_y = np.array(train_y)

    # val_x ,y = extract.extract_content_labels("./data/val")
    # val_x = model.tokenize_sequences(val_x)

    # val_y = []
    # for emotion in y:
    #     val_y.append(utils.get_one_hot_encoded_array_for_label(emotion,label_dict,label_map))

    # val_y = np.array(val_y)

    # test_x ,y = extract.extract_content_labels("./data/test")
    # test_x = model.tokenize_sequences(test_x)

    # test_y = []
    # for emotion in y:
    #     test_y.append(utils.get_one_hot_encoded_array_for_label(emotion,label_dict,label_map))

    # test_y = np.array(test_y)
    # ---------------------------------------------------------------------------------------------
    label_map = extract.load_label_map('./data/text_emotion.csv')

    X, y = extract.load_data_set('./data/text_emotion.csv')

    # init tokenizer using whole dataset
    model.init_tokenizer(X)
    
    # split the dataset into 80% training, 10% validation and 10% testing
    _80 = int(len(X)*.8)
    _10 = int(len(X)*.1)

    train_x = X[:_80]
    tr_y = y[:_80]

    val_x = X[_80:_80 + _10]
    v_y = y[_80:_80 + _10]

    test_x = X[80 + _10:]
    t_y = y[_80 + _10:]

    # tokenize the sentences
    train_x = model.tokenize_sequences(train_x)
    val_x = model.tokenize_sequences(val_x)
    test_x = model.tokenize_sequences(test_x)

    # create one hot vectors for the labels.
    train_y = []
    for emotion in tr_y:
        train_y.append(utils.get_one_hot_encoded_array(emotion,label_map))
    train_y = np.array(train_y)

    val_y = []
    for emotion in v_y:
        val_y.append(utils.get_one_hot_encoded_array(emotion,label_map))
    val_y = np.array(val_y)

    test_y = []
    for emotion in t_y:
        test_y.append(utils.get_one_hot_encoded_array(emotion,label_map))
    test_y = np.array(test_y)


    # ---------------------------------------------------------------------------------------------
    model.model()
    # Train the model
    history = model.train(train_x,train_y,(val_x,val_y),num_epochs=10)

    # Plot Training results
    model.plot_graphs(history,"accuracy")
    model.plot_graphs(history,"loss")

    # Test Results
    results = model.test(test_x,test_y)
    print("test loss, test acc:", results)

    # model.model.save('./model/my_model.h5')  # creates a HDF5 file 'my_model.h5'
    # del model  # deletes the existing model

    # # returns a compiled model
    # # identical to the previous one
    # model = load_model('my_model.h5')

    # # print(pad)