# Importing all the packages 
import pandas as pd
import numpy as np
import time

import sklearn
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
#importing keras packages
from keras import layers
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.preprocessing.text import Tokenizer, one_hot
from keras.preprocessing.sequence import pad_sequences
from keras import regularizers
from keras.metrics import categorical_accuracy
from keras.utils import np_utils
from keras.optimizers import Adam, Nadam, sgd, RMSprop, Adagrad

import matplotlib.pyplot as plt
#!pip install talos
#import talos as ta

#importing tensorflow packages
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Activation, Conv2D, Input, Embedding, Reshape, MaxPool2D, Concatenate, Flatten, Dropout, Dense, Conv1D
from tensorflow.keras.layers import MaxPool1D, MaxPooling1D
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.optimizers import Adam
import keras.preprocessing.text as kpt
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D
#from tensorflow.keras.layers import Conv2D, GlobalMaxPooling2D

#Path for the files to load from workspace

path_to_data = '/content/drive/My Drive/Test _data/tweets.json'
path_to_labels = '/content/drive/My Drive/Test _data/labels-train+dev.tsv'        
path_to_test = '/content/drive/My Drive/Test _data/labels-test.tsv'

#Load the tweet file and the label train files from the repository 
tweets = pd.read_json(path_to_data, orient='columns', lines=True, encoding='utf-8', dtype='int64')
tweets.rename(columns = {0:'Tweet_ID', 1:'Tweet'}, inplace=True)

#Load the development file and test path for processing and storing it in dataframe 
label_train= pd.read_csv(path_to_labels, sep='\t', names=['Label', 'tweet_id'])
test_data = pd.read_csv(path_to_test , sep='\t', names=['Label', 'tweet_id'])

#Normalizing the source dataframe by joining the data from the label based on tweet_id columns
merged_pd = pd.merge(left=tweets,right=label_train, left_on='Tweet_ID', right_on='tweet_id', how = 'inner')

#Filering the data for labels with less than 10 data training set.
final_df = merged_pd[['Tweet_ID','Label','Tweet']].groupby('Label').filter(lambda x: x['Label'].count()>9).reset_index()
merged_pd2 = pd.merge(left=tweets,right=test_data, left_on='Tweet_ID', right_on='tweet_id')
test_df1 = merged_pd2[['Tweet_ID','Label','Tweet']]

m = test_df1.Label.isin(final_df.Label)
test_df = test_df1[m]

#For the train and validation set
X = final_df['Tweet'].fillna('').tolist()
y = final_df['Label'].fillna('').tolist()

#For the final test set
X_ft = test_df['Tweet'].fillna('').tolist()
y_ft = test_df['Label'].fillna('').tolist()

# Split train & test
text_train, text_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# Tokenize and transform to integer index
tokenizer = Tokenizer(num_words=None,lower=True, char_level=True)
tokenizer.fit_on_texts(text_train)

X_train = tokenizer.texts_to_sequences(text_train) #train
X_test = tokenizer.texts_to_sequences(text_test)#validation set
X_ft = tokenizer.texts_to_sequences(X_ft) #finaltest set
vocab_size = len(tokenizer.word_index) + 1  # Adding 1 because of reserved 0 index
maxlen = max(len(x) for x in X_train) # longest text in train set

# Add pading to ensure all vectors have same dimensionality
X_train = pad_sequences(X_train, padding='post', maxlen=maxlen)
X_test = pad_sequences(X_test, padding='post', maxlen=maxlen)
X_ft = pad_sequences(X_ft, padding='post', maxlen=maxlen)

label_encoder = LabelEncoder()
# Converting the label encoding to one hot vectorized form
y_train = np_utils.to_categorical(label_encoder.fit_transform(y_train))
y_test = np_utils.to_categorical(label_encoder.transform(y_test))
y_ft = np_utils.to_categorical(label_encoder.transform(y_ft))

#### TALOS functionality for best parameter scan. Code is been commented as it requires high computational re 

# import talos as ta

# p = {  #'lr': (0.1,10,5),
#        #'batch_size': [50,100],
#        'dropout': (0, 0.4, 3),
#        'optimizer': ['adam', 'nadam'],
#        'strides':[1,2,3],
#        'filters':[128,256],
#        'kernel_size':[3,4,5]}

# embedding_dims = 50
# hidden_dims = 256

# def mymodel(X_train, y_train, x_val, y_val, params):
#   model = Sequential()
#   model.add(Embedding(vocab_size,embedding_dims,input_length=maxlen))

#   model.add(Conv1D(params['filters'],params['kernel_size'],padding='valid',activation='relu',strides= params['strides'], kernel_regularizer = regularizers.l2(0.001)))
#   model.add(Dropout(params['dropout']))
#   model.add(GlobalMaxPooling1D())
#   model.add(Dense(hidden_dims, kernel_regularizer = regularizers.l2(0.001)))
#   model.add(Activation('sigmoid'))
#   model.add(Dropout(params['dropout']))
#   model.add(Dense(y_train.shape[1]))
#   model.add(Activation('softmax'))
#   model.compile(loss='categorical_crossentropy',optimizer=params['optimizer'],metrics=["accuracy"])
#   history = model.fit(X_train, y_train,batch_size=100,epochs=5, validation_data=(X_test, y_test), shuffle=True, verbose=1)
#   return history, model

# h = ta.Scan(X_train, y_train,params=p,experiment_name='ad',model=mymodel,x_val=X_test, y_val=y_test)

#running on best parameters after hyperparameter-scanning again.
embedding_dims = 50
filters = 256
kernel_size = 5
hidden_dims = 256


model = Sequential()
model.add(Embedding(vocab_size,embedding_dims,input_length=maxlen))

model.add(Conv1D((filters),(kernel_size),padding='valid',activation='relu',strides=(1), kernel_regularizer = regularizers.l2(0.001)))
model.add(Dropout(0.20))
model.add(GlobalMaxPooling1D())
model.add(Dense(hidden_dims, kernel_regularizer = regularizers.l2(0.001)))
model.add(Activation('sigmoid'))
model.add(Dropout(0.20))
model.add(Dense(y_train.shape[1]))
model.add(Activation('softmax'))
adam = Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=["accuracy"])
print(model.summary())

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)

#Training model
history = model.fit(X_train, y_train,batch_size=100,epochs=40, validation_data=(X_test, y_test), shuffle=True, verbose=1, callbacks=[es])
#Test accuracy 
loss, accuracy = model.evaluate(X_ft, y_ft, verbose=1) 

#Confusion matrix
Y_pred = model.predict_classes(X_ft)
LABELS = final_df.Label.unique().tolist()
cnf_matrix = confusion_matrix(Y_pred, np.argmax(y_ft,axis=1))
plt.matshow(cnf_matrix, cmap = plt.cm.gray)


# Classification Report with F1 score
print(classification_report(y_ft, Y_pred, target_names=LABELS))