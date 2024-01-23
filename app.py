# Library imports
import pandas as pd
import numpy as np
import tensorflow as tf
# import re
from numpy import array

# from tensorflow.keras.preprocessing import one_hot, Tokenizer
from tensorflow.keras.models import load_model
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import Activation, Dropout, Dense
# from tensorflow.keras.layers import Flatten, GlobalMaxPooling1D, Embedding, Conv1D, LSTM
# from sklearn.model_selection import train_test_split
from flask import Flask, request,  render_template
from tensorflow.keras.preprocessing.sequence import pad_sequences

# import nltk
# nltk.download('stopwords')
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# import io
import json

# stopwords_list = set(nltk.corpus.stopwords.words('english'))
maxlen = 100

# Load model
model_path ='c1_lstm_model_acc_0.864.h5'
pretrained_lstm_model = load_model(model_path)

# Loading
with open('b3_tokenizer.json') as f:
    data = json.load(f)
    loaded_tokenizer = tokenizer_from_json(data)


# Create the app object
app = Flask(__name__)


# creating function for data cleaning
# from b2_preprocessing_function import CustomPreprocess
# custom = CustomPreprocess()


# Define predict function
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    query_asis = [str(x) for x in request.form.values()]
#     query_list = []
#     query_list.append(query_asis)
    
    # Preprocess review text with earlier defined preprocess_text function
    query_processed_list = []
    for query in query_asis:
        # query_processed = custom.preprocess_text(query)
        query_processed_list.append(query)
        
    # Tokenising instance with earlier trained tokeniser
    query_tokenized = loaded_tokenizer.texts_to_sequences(query_processed_list)
    
    # Pooling instance to have maxlength of 100 tokens
    query_padded = pad_sequences(query_tokenized, padding='post', maxlen=maxlen)
    
    # Passing tokenised instance to the LSTM model for predictions
    query_sentiments = pretrained_lstm_model.predict(query_padded)
    

    if query_sentiments[0][0]>0.5:
        return render_template('index.html', prediction_text=f"Positive Review with probable IMDb rating as: {np.round(query_sentiments[0][0]*10,1)}")
    else:
        return render_template('index.html', prediction_text=f"Negative Review with probable IMDb rating as: {np.round(query_sentiments[0][0]*10,1)}")


if __name__ == "__main__":
    app.run(debug=True)
