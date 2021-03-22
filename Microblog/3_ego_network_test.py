# Seed value
seed_value= 7

# Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.random.set_seed(seed_value)

# gpus = tf.config.list_physical_devices('GPU')
# tf.config.set_visible_devices(gpus[0], 'GPU')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

import pandas as pd
from  tensorflow.keras import Sequential
from sklearn.model_selection import train_test_split
from tensorflow_addons.layers import MultiHeadAttention
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import Constant, RandomNormal
from tensorflow.keras.layers import Layer, concatenate, Convolution1D, Embedding, Dense, Flatten, MaxPooling1D, Input, Bidirectional, LSTM, GlobalAveragePooling1D, LayerNormalization, Dropout

class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.8):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(head_size=embed_dim, num_heads=num_heads)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim),]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att([inputs, inputs])
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(Layer):
    def __init__(self, maxlen, vocab_size, embed_dim, weights):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = Embedding(input_dim=vocab_size, output_dim=embed_dim, embeddings_initializer=weights)
        self.pos_emb = Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions
    
def run(task, input_concat, model_type, i):
    
    '''
    task: string
        binary: Dataset with sentiment of postive and negative.
        multiclass: Dataset with sentiment of positive, neutral, and negative.
        
    input_concat: boolean
        True: The input is concatenated.
        False: The input is seperated.
    
    model_type: string
        CNN
        BiLSTM
        Transformer
        
    '''
    
    ### Dataset selection
    
    if task == 'multiclass':
        # dataset for multiclass
        df = pd.read_csv('data/geo_microblog.csv')
    else:
        # dataset for binary
        df = pd.read_csv('data/geo_microblog.csv')
        df = df[df.sentiment != 1]
        df.sentiment.replace(2, 1, inplace=True) # neutral

    ### Text processing
    
    # prepare tokenizer
    
    t = Tokenizer()
    t.fit_on_texts(df['text'])
    vocab_size = len(t.word_index) + 1
    
    # integer encode the documents
    
    encoded_docs = t.texts_to_sequences(df['text'])
    txtlen = 30
    loclen = 3
    
    # pad documents to a max length 
    
    padded_docs = pad_sequences(encoded_docs, txtlen, padding='post')

    ### Location processing
    
    loc = df[df.columns[-3:]]
    location = loc.values
        
    ### Prepare train and test set
        
    # merge txt and loc
    
    merge = np.concatenate((padded_docs, location),axis=1)
    
    # divide dataset to train and test set
    
    x_train, x_test, y_train, y_test = train_test_split(merge, df['sentiment'], test_size=0.3, random_state=100)
        
    if input_concat == False:
        # split train set to text and location
        x_train1 = x_train[:,:txtlen]
        x_train2 = x_train[:,-loclen:]
        
        # # split test set to text and location
        x_test1 = x_test[:,:txtlen]
        x_test2 = x_test[:,-loclen:]
        
    ### Pretrained word embedding
    
    # load the whole embedding into memory
    
    #embeddings_index = dict()
    #f = open('../glove.twitter.27B.200d.txt')
    #for line in f:
    #	values = line.split()
    #	word = values[0]
    #	coefs = asarray(values[1:], dtype='float32')
    #	embeddings_index[word] = coefs
    #f.close()
    #print('Loaded %s word vectors.' % len(embeddings_index))
    
    # create a weight matrix for words in training docs
    
    vector_dimension = 200
    #embedding_matrix = zeros((vocab_size, vector_dimension))
    #for word, i in t.word_index.items():
    #	embedding_vector = embeddings_index.get(word)
    #	if embedding_vector is not None:
    #		embedding_matrix[i] = embedding_vector
    
    if task == 'binary':
        embedding_matrix = np.load('variable/microblog_binary_embedding_matrix.npy')
    else:
        embedding_matrix = np.load('variable/microblog_multiclass_embedding_matrix.npy')
    
    ### Deep Learning model
    
    if input_concat == True:
        input_dimension = txtlen+loclen
        inputs = Input(shape=(input_dimension,))
        embedding_layer = Embedding(vocab_size, vector_dimension, embeddings_initializer=Constant(embedding_matrix), input_length=input_dimension)(inputs)
    else:
        inputText = Input(shape=(txtlen,))
        x = Embedding(vocab_size, vector_dimension, embeddings_initializer=Constant(embedding_matrix), input_length=txtlen)(inputText)
        inputLocation = Input(shape=(loclen,))
        y = Embedding(vocab_size, vector_dimension, embeddings_initializer=RandomNormal(), input_length=loclen)(inputLocation)
        embedding_layer = concatenate([x, y], axis=1)
    
    if model_type == "CNN":
        # CNN
        convolution_first = Convolution1D(filters=100, kernel_size=5, activation='relu')(embedding_layer)
        convolution_second = Convolution1D(filters=100, kernel_size=4, activation='relu')(convolution_first)
        convolution_third = Convolution1D(filters=100, kernel_size=3, activation='relu')(convolution_second)
        pooling_max = MaxPooling1D(pool_size=2)(convolution_third)
        flatten_layer = Flatten()(pooling_max)
        dense = Dense(20, activation="relu")(flatten_layer)
        if task == 'binary':
            outputs = Dense(units=1, activation='sigmoid')(dense)
        else:
            outputs = Dense(units=3, activation='softmax')(dense)
    
    if model_type == "BiLSTM":
        ### BiLSTM
        lstm_first = Bidirectional(LSTM(units=100))(embedding_layer)
        dense = Dense(20, activation="relu")(lstm_first)
        if task == 'binary':
            outputs = Dense(1, activation='sigmoid')(dense)
        else:
            outputs = Dense(3, activation='softmax')(dense)
            
    if model_type == "Transformer":
        ### Transformer
        num_heads = 2  # Number of attention heads
        ff_dim = 32  # Hidden layer size in feed forward network inside transformer
        
        if input_concat == True:
            embedding_layer_weighted = TokenAndPositionEmbedding(input_dimension, vocab_size, vector_dimension, Constant(embedding_matrix))
            x = embedding_layer_weighted(inputs)
        else:
            embedding_layer_weighted = TokenAndPositionEmbedding(txtlen, vocab_size, vector_dimension, Constant(embedding_matrix))
            x = embedding_layer_weighted(inputText)
            embedding_layer = TokenAndPositionEmbedding(loclen, vocab_size, vector_dimension, RandomNormal())
            y = embedding_layer(inputLocation)
            x = concatenate([x, y], axis=1)
        
        transformer_block = TransformerBlock(vector_dimension, num_heads, ff_dim)
        x = transformer_block(x)
        x = GlobalAveragePooling1D()(x)
        x = Dense(20, activation="relu")(x)
        if task == 'binary':
            outputs = Dense(1, activation='sigmoid')(x)
        else:
            outputs = Dense(3, activation='softmax')(x)
                
    
    # build model
    
    if input_concat == True: 
        model = Model(inputs, outputs)
    else:
        model = Model(inputs=[inputText, inputLocation], outputs=outputs)
    
    # compile model
    
    if task == 'binary':
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    else:
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['sparse_categorical_accuracy'])    

        
    if input_concat == True:
        
        # load model 
                    
        model.load_weights(f"saved/{task}/3_ego/concat/{model_type}/{task}_ego_concat_{model_type}_{i}.h5")
         
        # evaluate model
        
        loss, accuracy = model.evaluate(x_test, y_test, verbose=1)
        
    else:
        
        # load model 
        
        model.load_weights(f"saved/{task}/3_ego/both/{model_type}/{task}_ego_both_{model_type}_{i}.h5")
        
        # evaluate model
        loss, accuracy = model.evaluate([x_test1, x_test2], y_test, verbose=1)
    
    return loss, accuracy

### Run all

task_list = ['binary'] 
input_concat_list = [True, False]
model_type_list = ["CNN", "BiLSTM", "Transformer"]
repeat_num = 10
df_new = pd.DataFrame(columns=["accuracy", "loss"])

for task in task_list:
    for input_concat in input_concat_list:
        for model_type in model_type_list:
            acc, ls = [], []
            for i in range(repeat_num):
                loss, accuracy= run(task, input_concat, model_type, i)
                
                acc.append(accuracy)
                ls.append(loss)
            
            df_temp = pd.DataFrame(columns=["accuracy", "loss"], index=range(repeat_num))
            df_temp["accuracy"] = acc
            df_temp["loss"] = ls
            df_temp.at[repeat_num, "accuracy"] = f"{task}_concat_{input_concat}_{model_type}"
            df_new = df_new.append(df_temp)

df_new.to_csv("saved/3_ego_binary.csv", index=False)
        
### Single Run
# task = 'multiclass'
# input_concat = False
# model_type = "Transformer"
# repeat_num = 10                

# model, history, loss, accuracy, max_loss, min_loss, avg_loss, max_accuracy, min_accuracy, avg_accuracy = \
#     run(task, input_concat, model_type, repeat_num)

# visualize model
# tf.keras.utils.plot_model(model, show_shapes=True)
# reconstructed_model = keras.models.load_model("my_h5_model.h5")



