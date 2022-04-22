
## Text Categorisation of the 20 NewsGroups data

#Date: April 2021

#Overview

'''
The Project contains four Python Notebooks designed to run on
Google Colab.

1.0 Preprocessing

2.0 Analysis of 20 NewsGroups Data.ipynb

3.0 Simple Models (Chen code)

4.0 Advanced Models (Gideon code)

'''

#######################################################################
## Connection setup

# you should accept the connection between Drive and the code 

from google.colab import drive
drive.mount('/content/drive')

# give your project directory here. The unzipped 20NewsGroups trainng and test
data sets should be in this location
proj_dir='/content/drive/MyDrive/Colab Notebooks/doc2vec/' 

#Ensure the GPU runtime is selected when running code 3.0 and 4.0 

#######################################################################

1.0 Data Preprocessing

#######################################################################

# Description:  
'''
1. Import data

2. Clean data as follows:

Lower the english letters
Remove headers
Drop all digits
Remove URLs and Emails
Drop all punctuation from our text
Drop stop words
Lemmatize words

3. Vectorise data

'''
# Installation:

import os
import pandas as pd
import numpy as np
import collections
import re
import string
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
nltk.download('wordnet')

# Usage:

```

## Load and clean the data

'''this function is used to read text files.'''
def read_txt_file(file_name):


#load stop words
stop_words=set(stopwords.words('english'))


#get punctuations
punctuations=string.punctuation


'''this function is used to clean text'''
def clean_txt(txt):



'''this function is perform loading and cleaning'''
load_and_clean_data(location): 


#save clean data sets before vectorize it 
df_train.to_csv(proj_dir+'cleaned_train_data.csv')
df_test.to_csv(proj_dir+'cleaned_test_data.csv')

```

## Convert to vectors using doc2vec

'''this function is used to do tokenization'''
def tokenizer(txt): 

#split data into train and validation sets
train_,validation=train_test_split( train, test_size=0.33, random_state=42) 


#save data sets after vectorize it 
train_.to_csv(proj_dir+'train_data.csv')
test.to_csv(proj_dir+'test_data.csv')
validation.to_csv(proj_dir+'validation_data.csv')



```
## Modeling

#read data which was saved in the feature generation part
train=pd.read_csv(proj_dir+'train_data.csv')
test=pd.read_csv(proj_dir+'test_data.csv')
validation=pd.read_csv(proj_dir+'validation_data.csv')

# Category encoding
# keras.utils.to_categorical is a function used to Converts a class vector (integers) to binary class matrix
# so by the end of this code block we will have encoded categories so we are ready to use it in the model. 
y_train_en=keras.utils.to_categorical(y_train_en)
y_test_en=keras.utils.to_categorical(y_test_en)
y_validation_en=keras.utils.to_categorical(y_validation_en)


#######################################################################

2.0 Data Analysis

#######################################################################

## Description:

'''Perform data analysis of the cleaned 20NewsGroups training data set
1) Summary Statistics of Data

2) Most common words for each category

3) Analysis of Parts of Speech

4) Bigram analysis

data is imported as csv after text cleaning, but pre-vectorising.

'''

## Installation: 
install pandas
install numpy
install collections
install matplotlib
install re
install nltk

## Usage:
```
## load cleaned data
'''ensure 1.0 Data Preprocessing.ipynb has been run'''

train_cleaned_df = pd.read_csv(proj_dir+'train_cleaned_data.csv')

```

```
## Summary Statistics of Data

## Most Common words for each category

#Installation
from collections import Counter
import matplotlib.pyplot as plt
import re

#Description
''' count total of each word '''
d = Counter(vocab)

'''Creates bar chars for each category dsiplaying 10 most common words'''
plt.bar(words, counts)

```

```
## Analysis of Parts of Speech
''' Identify the number of nouns and verbs etc in each group and display as a stacked bar chart '''

#Counting words
''' this function counts the number of nouns '''
def NounCount(x):

''' this function counts the number of verbs'''
def VerbCount(x):

''' this function counts thenumber of adverbs'''
def AdverbCount(x):

''' this function counts the number of adjectives'''
def AdjectiveCount(x):

''' this function counts the number of words which are neither nouns, adverbs, verbs or adjectives
def OtherCount(x):

#Visualisatons
''' visualisation to show split of word types'''
ax = train_cl_gr_df.plot.barh(stacked=True,  title='POS Categorisation', x='folder name')

'''Show the POS values as the proportion of the total number of words for each category''' 
ax = train_pos_df_scaled.plot.barh(stacked=True,  title='POS Categorisation (Scaled)', x='folder name')

```



## Bigram Analysis
from nltk import bigrams
import matplotlib.pyplot as plt

'''This section calculates the most frequently occuring bigrams'''


# plot most frequent bigrams in whole text
'''calculate frequency of bigrams using bigrams from nltk'''

#Calculate Pointwise mutiual information'''
# Using BigramAssocMeasures, BigramCollocationFinders




#######################################################################

3.0 Basic Models (Chen code)

#######################################################################


# Description: (what the code does)
'''
1. Combine validation and training sets to do grid search

2. Grid search and model training of naive bayes

3. Evaluate naive bayes performance on test set through accuracy, F1 score, recall, precision and confusion matrix

4. Grid search and model training of KNN

5. Evaluate KNN performance on test set through accuracy, F1 score, recall, precision and confusion matrix

6. Build and visualize 1 dimensional CNN without using Pre-trained Embedding

7. Exhibit Accuracy and Validation Loss of CNN Model

8. Evaluate CNN performance on test set through accuracy, F1 score, recall, precision and confusion matrix


'''
# Installation: (which packages to install)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras import initializers
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Layer, InputSpec, TextVectorization
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding, MaxPooling1D,GRU, Bidirectional
from keras.utils.vis_utils import plot_model


# Usage:  (description of code and functions)

# combine validation and training sets to do grid search
X_train_val = pd.concat([X_train, X_validation], axis = 0)
X_train_val = X_train_val.reset_index(drop = True)
y_train_val = np.concatenate((y_train_en, y_validation_en))
y_train_val  = pd.DataFrame(y_train_val).apply(lambda x: x.argmax(), axis=1).values

# grid search of Naive Bayes
gnb_clf = GaussianNB()
parameters = {
    'var_smoothing': [1e-1,1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15]
}
clf = GridSearchCV(gnb_clf, parameters, cv=pds, verbose=1, n_jobs=1) #it is hold-out validation here
clf.fit(X_train_val, y_train_val)

#report Naive Bayes performance on test set
nb_pred = clf.best_estimator_.predict(X_test)
macro_averaged_precision = precision_score(y_test, nb_pred, average = 'macro')
recall = recall_score(y_test, nb_pred, average = 'macro')
macro_averaged_f1 = f1_score(y_test, nb_pred, average = 'macro')
cm = confusion_matrix(y_test, nb_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=names)

# grid search KNN
k_range = list(range(1, 31)) # possible k
param_grid = dict(n_neighbors=k_range)
knn = KNeighborsClassifier()
clf = GridSearchCV(knn, param_grid, cv=pds, verbose=1, n_jobs=1)
clf =clf.fit(X_train_val, y_train_val)

#report KNN performance on test set
knn_pred = clf.best_estimator_.predict(X_test)
macro_averaged_precision = precision_score(y_test, knn_pred, average = 'macro')
recall = recall_score(y_test, knn_pred, average = 'macro')
macro_averaged_f1 = f1_score(y_test, knn_pred, average = 'macro')
cm = confusion_matrix(y_test, nb_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=names)

# Creat Text Vectorization from Training Dataset
vectorizer = TextVectorization(max_tokens=20000, output_sequence_length=200)
text_ds = tf.data.Dataset.from_tensor_slices(X_train.values.tolist())
vectorizer.adapt(text_ds)
voc = vectorizer.get_vocabulary()

# Prepare Input Data for CNN Model
x_train = vectorizer(np.array([[s] for s in X_train.values.tolist()])).numpy()
x_val = vectorizer(np.array([[s] for s in X_val.values.tolist()])).numpy()
x_test = vectorizer(np.array([[s] for s in X_test.values.tolist()])).numpy()
y_train=np.array(np.unique(y_train, return_inverse=True)[1])
y_val=(np.unique(y_val, return_inverse=True)[1])
y_test=(np.unique(y_test, return_inverse=True)[1])

# Build CNN model
inputs = tf.keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(num_tokens, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
predictions = layers.Dense(len(label_names), activation="softmax", name="predictions")(x)
model = tf.keras.Model(inputs, predictions)
model.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
print("Simple Neural Network")
model.summary()

# Evaluate model performance on test set
_, test_accuracy = model.evaluate(x=x_test, y=y_test, verbose=0)

#######################################################################

4.0 Advanced Models (Gideon code)

#######################################################################


# Description: (what the code does)

# Installation: (which packages to install)

# Usage:  (description of code and functions)


#######################################################################
## License:
The 20NewsGroup data is open source
All code owned by Cardiff University


### END ###