import numpy as np
import pandas as pd 
import os
import matplotlib.pyplot as plt
import sklearn.feature_extraction.text as sk_text
import json
import csv


# from collections.abc import Sequence
from sklearn import preprocessing
# import shutil

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation
import io 
from sklearn import metrics
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint


# the original data is stored in review.json
# there is also a business.json file which has all the business ratings/names/business_ids
#
#
#
# FROM ASSIGNMENT 
outfile = open('review_stars.tsv','w')
sfile = csv.writer(outfile, delimiter='\t', quoting = csv.QUOTE_MINIMAL)
sfile.writerow(['business_id','stars','text'])

with open('review.json', encoding='utf-8') as f:
    for line in f:
        row = json.loads(line)
        # some special char must be encoded in utf-8
        sfile.writerow([row['business_id'], row['stars'], (row['text']).encode('utf-8')])
outfile.close()

# here is the initial dataframe for reviews named df and the business dataframe is named business
# dropping these columns but may want to keep state and city for looking at those individual areas later on. ...
df = pd.read_csv('review_stars.tsv', delimiter='\t', encoding='utf-8')
business = pd.read_json('business.json', lines=True) 
business.drop(['address','city','state','postal_code','latitude','longitude', 'review_count', 'is_open', 'attributes', 'categories', 'hours'], axis=1, inplace=True)


df_review_agg = df.groupby('business_id')['text'].sum()
df_ready_for_sklearn = pd.DataFrame({'business_id':df_review_agg.index, 'all_reviews':df_review_agg.values})
df3 = pd.merge(df_ready_for_sklearn, business, on="business_id")
df3.drop(['business_id','name'],axis=1,inplace=True)
# df3 just contains reviews and the stars associated with each business



# model stuff
vectorizer = sk_text.CountVectorizer(min_df=1)
corpus = df3['all_reviews']
vectorizer = sk_text.TfidfVectorizer(max_features = 5000,min_df=5)
matrix = vectorizer.fit_transform(corpus)
# x = tfidf_data
x = matrix.toarray()
y = df3.iloc[:,1].values

#
#
#    
#
#
#

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.25, random_state=42)
# first hidden last has to have input matching the dimension of a row in x

checkpointer = ModelCheckpoint(filepath="dnn/best_weights.hdf5", verbose=0, save_best_only=True)

# for loop to re-initialize models to avoid local optimum

for i in range(5):
    model = Sequential()
    model.add(Dense(25, input_dim = x.shape[1], activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=2, verbose=2, mode='auto')
    model.fit(x_train, y_train, validation_data=(x_test, y_test), callbacks=[monitor, checkpointer], verbose=2, epochs=10)


# now to 
pred = model.predict(x_test)
rmse = np.sqrt(metrics.mean_squared_error(pred,y_test))
print(rmse)


plt.figure(figsize = (14,10))
sns.set(palette = "viridis")
sns.set_style("white")
sns.distplot(data = data["test"],
             label = "test values",
             kde = False,
             bins = 500, # can play with this 
             hist_kws = {"edgecolor" : 'none', "alpha" : 0.7 })
sns.distplot(data = data["pred"],
             label = "test values",
             kde = False,
             bins = 500, # can play with this 
             hist_kws = {"edgecolor" : 'none', "alpha" : 0.7 })
plt.legend(["Actual Values", "Predicted Values"])
plt.show()




