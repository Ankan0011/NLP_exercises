#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier


#Path for the files to load from workspace

path_to_data = 'C:/Users/Ankan/Documents/University docs/NLP/exercise_1/tweets.json'
path_to_labels = 'C:/Users/Ankan/Documents/University docs/NLP/exercise_1/labels-train+dev.tsv'        
path_to_test = 'C:/Users/Ankan/Documents/University docs/NLP/exercise_1/labels-test.tsv'

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

#Stratified Shuffle split for spliting data for training and testing in 90-10 % proportion.
split = StratifiedShuffleSplit(test_size= 0.1 , random_state=42)
for train_index, test_index in split.split(final_df, final_df['Label']):
    strat_train_set = final_df.loc[train_index]
    strat_test_set = final_df.loc[test_index]

X_train = strat_train_set.Tweet.astype(str)
y_train = strat_train_set.Label.astype(str)
X_test =  strat_test_set.Tweet.astype(str)
y_test =  strat_test_set.Label.astype(str)
X_test_final = test_df.Tweet
y_test_final = test_df.Label

#This is what we got here
print('Training set shape: ', X_train.shape)
print('Training labels shape: ', y_train.shape)
print('Test set shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
print('Final test set shape: ', X_test_final.shape)
print('Final test label shape: ', y_test_final.shape)

#%%

#LabelEncoder for encoding the Y_train & y_ytest
label_encoder = LabelEncoder()
label_encoder_final = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

label_encoder_final.fit_transform(final_df.Label)
y_test_final1 = label_encoder.transform(y_test_final)


pipe_MLP = Pipeline([
    ('vect', CountVectorizer(ngram_range = (1,3))),
    ('tfidf', TfidfTransformer()),
    ('MLP_clf', MLPClassifier(hidden_layer_sizes = (50,), max_iter = 10, alpha = 0.003))
])

pipe_MLP.fit(X_train, y_train)
score = pipe_MLP.score(X_train, y_train)
score_final = pipe_MLP.score(X_test_final, y_test_final1)
print('MLPClassifier pipeline training accuracy :', score)
print('MLPClassifier pipeline test accuracy :', score_final)

#%%
param_grid = { 'vect__ngram_range': [(1,3),(2,3)],
               'vect__analyzer': ['char','char_wb'],
               'MLP_clf__hidden_layer_sizes': [(25,1),(25,3)],
               'MLP_clf__activation' : ['logistic', 'tanh', 'relu'],
               'MLP_clf__solver': ['lbfgs','adam', 'sgd'],
               'MLP_clf__momentum' : [0.2,0.7, 0.95],
               'MLP_clf__n_iter_no_change':[50, 25]
               }


gs_mlp = GridSearchCV(pipe_MLP, param_grid, cv=3, n_jobs=5, verbose=1)
gs_mlp.fit(X_train, y_train)

svc_df = pd.DataFrame.from_dict(gs_mlp.cv_results_)
svc_df.sort_values(by=["rank_test_score"])

print(precision_score(y_test_final1, gs_mlp.predict(X_test_final), average='micro'))
print(recall_score(y_test_final1, gs_mlp.predict(X_test_final), average='micro'))
print(f1_score(y_test_final1, gs_mlp.predict(X_test_final), average='micro'))

#%%

import matplotlib.pyplot as plt

conf_mx = confusion_matrix(y_test_final1, gs_mlp.predict(X_test_final))
plt.matshow(conf_mx, cmap = plt.cm.gray)
plt.show()
