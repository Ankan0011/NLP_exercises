#%%
import pandas as pd
import numpy as np
import json, os
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn import linear_model
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score, cross_val_predict, GridSearchCV
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt

#%%
# read data from .csv file by using absolute path
# __file1__ = 'labels-train+dev.tsv'
# __file2__ = 'tweets.json'
# my_absolute_dirpath = os.path.abspath(os.path.dirname(__file1__))
#df1 = pd.read_csv(my_absolute_dirpath+'/' + __file1__, sep=',')



#Path for the files

path_to_data = 'C:/Users/Ankan/Documents/University docs/NLP/exercise_1/tweets.json'
path_to_labels = 'C:/Users/Ankan/Documents/University docs/NLP/exercise_1/labels-train+dev.tsv'        
path_to_test = 'C:/Users/Ankan/Documents/University docs/NLP/exercise_1/labels-test.tsv'

#Load the tweet file and the label train files from the repository 
tweets = pd.read_json(path_to_data, orient='columns', lines=True, encoding='utf-8', dtype='int64')
tweets.rename(columns = {0:'Tweet_ID', 1:'Tweet'}, inplace=True)
label_train= pd.read_csv(path_to_labels, sep='\t', names=['Label', 'tweet_id'])
test_data = pd.read_csv(path_to_test , sep='\t', names=['Label', 'tweet_id'])


merged_pd = pd.merge(left=tweets,right=label_train, left_on='Tweet_ID', right_on='tweet_id', how = 'inner')
final_df = merged_pd[['Tweet_ID','Label','Tweet']].groupby('Label').filter(lambda x: x['Label'].count()>10).reset_index()
merged_pd2 = pd.merge(left=tweets,right=test_data, left_on='Tweet_ID', right_on='tweet_id')
test_df1 = merged_pd2[['Tweet_ID','Label','Tweet']]

#test_df = test_df1[test_df1.set_index(['Label']).index.isin(final_df.set_index(['Label']).index)]
#test_df = test_df1[test_df1['Label'].isin(final_df['Label'])]
m = test_df1.Label.isin(final_df.Label)
test_df = test_df1[m]


#%%

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


#LabelBinizer for encoding the Y_train & y_test
# encoder = LabelBinarizer(sparse_output = True)
# y_train_f = encoder.fit_transform(y_train)
# y_test_f = encoder.transform(y_test)



#LabelEncoder for encoding the Y_train & y_ytest
label_encoder = LabelEncoder()
label_encoder_final = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

label_encoder_final.fit_transform(final_df.Label)
y_test_final1 = label_encoder.transform(y_test_final)

#print(y_test_final1)

#%%


#CountVectorizer for ngram range
count_vect = CountVectorizer(lowercase=True, ngram_range=(1,3))
X_train_counts = count_vect.fit_transform(X_train)
X_test_counts = count_vect.transform(X_test)


X_test_final =  count_vect.transform(X_test_final)

#%%

tfidf_tranformer = TfidfTransformer(smooth_idf=True ).fit(X_train_counts)
X_train_tfidf = tfidf_tranformer.transform(X_train_counts)
X_test_tfidf  = tfidf_tranformer.transform(X_test_counts)
X_test_tfidf_final  = tfidf_tranformer.transform(X_test_final)

#%%

# def display_scores(scores):
#     print("Scores:", scores)
#     print("Mean:", scores.mean())
#     print("Standard deviation:", scores.std())

#%%

clf = linear_model.SGDClassifier(max_iter=900, loss='hinge', random_state= 42)
#clf.fit(X_train_tfidf, y_train)
clf.fit(X_test_tfidf_final, y_test_final1)
# clf.predict(X_test_tfidf_final)
#scores_train = cross_val_score(clf, X_train_tfidf, y_train, scoring='accuracy', cv=3)
#scores_test = cross_val_score(clf, X_test_tfidf, y_test, scoring='accuracy', cv=3)

scores_final = cross_val_score(clf, X_test_tfidf_final, y_test_final1, scoring='accuracy', cv=3)
#lin_rmse_scores = np.sqrt(-scores_dev)
#display_scores(lin_rmse_scores)
# print("Train Score")
# print(scores_train)
# print("Test Scores")
# print(scores_test)
#print(clf.score(X_train_tfidf, y_train))
print(clf.score(X_test_tfidf_final, y_test_final1))



# clf.fit(X_test_tfidf_final, y_test_final1)
# y_train_predictions = cross_val_predict(clf, X_test_tfidf_final, y_test_final1, cv=3)
# print(precision_score(y_test_final1, y_train_predictions, average='micro'))
# print(recall_score(y_test_final1, y_train_predictions, average='micro'))
# print(f1_score(y_test_final1, y_train_predictions, average='micro'))
# conf_mx = confusion_matrix(y_test_final1, y_train_predictions)
# conf_mx

#%%


multiNB = MultinomialNB(fit_prior=False)
multiNB.fit(X_train_tfidf, y_train)
#multiNB.fit(X_test_tfidf_final, y_test_final1)


scores = cross_val_score(multiNB, X_test_tfidf_final, y_test_final1, scoring='accuracy', cv=4)
#print(scores)
print(multiNB.score(X_train_tfidf, y_train))
print(multiNB.score(X_test_tfidf_final, y_test_final1))
#%%

#%%

pipe_SGD = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('sgd_clf', linear_model.SGDClassifier())
])

pipe_SGD.fit(X_train, y_train)
score = pipe_SGD.score(X_train, y_train)
score_final = pipe_SGD.score(X_test_final, y_test_final1)
print('SGDClassifier pipeline training accuracy :', score)
print('SGDClassifier pipeline test accuracy :', score_final)

#%%


param_grid = { 'sgd_clf__loss': ['hinge', 'modified_huber','squared_hinge'],
             'sgd_clf__penalty': ['l2', 'elasticnet'],
             'tfidf__norm': ['l1', 'l2'], 'vect__analyzer': ['char','char_wb'],
             'vect__ngram_range': [(1,3),(2,3),(2,4)],
             'sgd_clf__class_weight':['balanced']
            # 'sgd_clf__n_iter_no_change':[]
             }

gs_svc = GridSearchCV(pipe_SGD, param_grid, cv=2, n_jobs=8, verbose=1)
gs_svc.fit(X_train, y_train)

#y_train_predictions = cross_val_predict(gs_svc, X_test_final, y_test_final1, cv=3)
print(precision_score(y_test_final1, gs_svc.predict(X_test_final), average='micro'))
print(recall_score(y_test_final1, gs_svc.predict(X_test_final), average='micro'))
print(f1_score(y_test_final1, gs_svc.predict(X_test_final), average='micro'))

svc_df = pd.DataFrame.from_dict(gs_svc.cv_results_)
svc_df.sort_values(by=["rank_test_score"])

conf_mx = confusion_matrix(y_test_final1, gs_svc.predict(X_test_final))
plt.matshow(conf_mx, cmap = plt.cm.gray)
plt.show()

#%%

#Pipeline for executing Multinomial NB for training on test data

pipe_MNB = Pipeline([
    ('vect_mnb', CountVectorizer()),
    ('nb_clf', MultinomialNB())
])

pipe_MNB.fit(X_train, y_train)
score_MNB = pipe_MNB.score(X_train, y_train)
score_final_MNB = pipe_MNB.score(X_test_final, y_test_final1)
print('MultinomialNB pipeline training accuracy :', score_MNB)
print('MultinomialNB pipeline test accuracy :', score_final_MNB)
#%%

param_grid = { 'vect_mnb__ngram_range': [(1,3),(2,3),(1,4),(2,4)],
               'nb_clf__fit_prior': [False],
               'vect_mnb__analyzer': ['char','char_wb']}

gs_mnb = GridSearchCV(pipe_MNB, param_grid, cv=3, n_jobs=10, verbose=1)
gs_mnb.fit(X_train, y_train)

svc_df = pd.DataFrame.from_dict(gs_mnb.cv_results_)
svc_df.sort_values(by=["rank_test_score"])

#scores_cvs = cross_val_predict(pipe_MNB, X_test_final, y_test_final1, scoring='accuracy', cv=3)

print(precision_score(y_test_final1, gs_mnb.predict(X_test_final), average='micro'))
print(recall_score(y_test_final1, gs_mnb.predict(X_test_final), average='micro'))
print(f1_score(y_test_final1, gs_mnb.predict(X_test_final), average='micro'))

conf_mx = confusion_matrix(y_test_final1, gs_mnb.predict(X_test_final))
plt.matshow(conf_mx, cmap = plt.cm.gray)
plt.show()

#%%

