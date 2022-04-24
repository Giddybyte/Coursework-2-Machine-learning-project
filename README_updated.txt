
## Text Categorisation of the 20 NewsGroups data

#Date: April 2021

#Overview

'''
The Project contains two Python Notebooks designed to run on
Google Colab.

1.0 1 Submission Preprocessing and Implementation.ipynb   (Preprocessing and Implementation)

2.0 Analysis of 20 NewsGroups Data.ipynb             (Data Analysis)


Code 1.0 consists of 3 sections. Pre-processing, Simple models and Advanced Models.

'''

#######################################################################
## Connection setup

# you should accept the connection between Drive and the code 

from google.colab import drive
drive.mount('/content/drive')

# give your project directory here. The unzipped 20NewsGroups trainng and test
data sets should be in this location
proj_dir='/content/drive/MyDrive/Colab Notebooks/doc2vec/' 

#Ensure the GPU runtime is selected when running code 1.0 when running Keras functions


# Alternaive Data Collection
'''
The data set can be obtained from many sources. The following is one method to download the data as a tar.gz
file which wil then need to be unzipped

'''

pip install wget
import wget
site_url = 'http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz'
file_name = wget.download(site_url)
#print(file_name)
import tarfile
my_tar = tarfile.open(file_name)
my_tar.extractall(path) # specify which folder to extract to
my_tar.close()



#######################################################################

NoteBook 1 Submission Preprocessing and Implementation.ipynb

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

1.2 Basic Models (Chen code)

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

1.3 Advanced Models (Gideon code)

#######################################################################


# Description: (what the code does)
First, we wrote a small utility to process the data directly from the given website.Thereafter I applied the proprocessing team processed training and testing dataset. First, I created  three neural network models one directly adapted from keras example and the other two more advanced, one dimension Convolution model CONVD1, with Max and global pooling layers and Bidirectional LSTM.

I started  with vectorizing the training dataset using Tensorflow Text Vectorization module,created and embedding layer using the vocalulary and pretrained matrix derived from Glove pretrained word vectors.

We fit the two neural network model to the training data(shuffled and split between training and validation set). We use callback and model checkpoint for resource management.The maximun epoch reached in both cases after the callback is ninteen(19). The accuracy and error where plotted and we got up to 78% from the bidirectional LSTM after lost of try with hyperparameter adjustment-vocabulary size,GloVe dimension,sequence lenght,embedding size, LSTM units. We got a 78% with 30,000 vocabulary, GloVe 200 word vector dimention, 200 embedding lenght and 300 units of LSTM.

# Installation: (which packages to install)
The following packages was used diretly in my codes. 
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.test.utils import get_tmpfile
from nltk.stem import WordNetLemmatizer
import nltk
from sklearn.model_selection import train_test_split
nltk.download('stopwords')
nltk.download('wordnet')
import nltk
nltk.download('punkt')
from __future__ import division, print_function
from gensim import models
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Reshape, Flatten, concatenate, Input, Conv1D, GlobalMaxPooling1D, Embedding
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Model
from sklearn.model_selection import train_test_split
from tensorflow import keras
import numpy as np
import pandas as pd
import os
import collections
import re
import string
import pickle
from collections import defaultdict
import re
from bs4 import BeautifulSoup
import sys
import os
os.environ['KERAS_BACKEND']='theano'
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Embedding
from keras.layers import Dense, Input, Flatten
from keras.layers import Conv1D, MaxPooling1D, Embedding, Dropout, LSTM, GRU, Bidirectional
from keras.models import Model
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from keras import backend as K
from tensorflow.keras.layers import Layer, InputSpec
import tensorflow as tf
from tensorflow import keras
from keras import initializers
%matplotlib inline
from sklearn.model_selection import StratifiedShuffleSplit
from google.colab import drive
pip install wget****must be installed first
import wget
import tarfile

# Usage:  (description of code and functions)


"""# **Creating the folder in colab to warehouse all subfolders(training and testing subfolders)** Please run this code only once to create folder otherwise:FileExistsError"""

import os
path = '/content/drive/MyDrive/Colab Notebooks/doc2vec6/'
os.mkdir(path)

"""# **Installing the module for downloading the tae.gz file from the website **"""

pip install wget

"""## **Collecting the zipped file from the website  and unzip into the created folder**. This create the required two folder training and testing


"""

import wget
site_url = 'http://qwone.com/~jason/20Newsgroups/20news-bydate.tar.gz'
file_name = wget.download(site_url)
#print(file_name)
import tarfile
my_tar = tarfile.open(file_name)
my_tar.extractall(path) # specify which folder to extract to
my_tar.close()

proj_dir=path #give your project directory here. data sets should be in this location

'''this function is used to read text files.'''
def read_txt_file(file_name):
    with open(file_name,encoding="utf8", errors='ignore') as f:
        ###extract the boady of the text###
        line = f.readline()
        txt=''
        txt=txt+' '+line
        while line:
            line = f.readline()
            txt=txt+' '+line
        ###################################
    f.close()
    return(txt)

stop_words=set(stopwords.words('english'))#load stop words
punctuations=string.punctuation #get punctuations
lemmatizer = WordNetLemmatizer()
'''this function is used to clean text'''
def clean_txt(txt):
    txt=txt.lower() #set all characters to lowercase
    sentences=txt.split('\n')
    txt = ' '.join([i for i in sentences if not ':' in i])#remove headers
    txt = ''.join([i for i in txt if not i.isdigit()])#remove numbers

    ###remove urls and emails###
    words=txt.split()
    txt = ' '.join([i for i in words if not '@' in i and not '.com' in i and not  'http:' in i])
    #######################################

    ###remove punctuations###
    for character in punctuations:
        txt = txt.replace(character, '')
    #########################################
    
    ###remove stop words and lemmatize###
    words=txt.split()
    filtered_txt = ' '.join([lemmatizer.lemmatize(i) for i in words if not i in stop_words])
    #####################################
    
    return(filtered_txt)

def load_and_clean_data(location):    
    y=os.listdir(location)#get the list of folder
    txts=[]
    txts_cleaned=[]
    folder_array=[]
    file_array=[]
    for i in range(len(y)):
        text_file_names=os.listdir(location+'/'+y[i]) #get the list of files
        for text_file_name in text_file_names:
                file_array.append(text_file_name)
                txt=read_txt_file(location+'/'+y[i]+'/'+text_file_name) #read the text file
                txts.append(txt)
                txts_cleaned.append(clean_txt(txt)) #clean the text
                folder_array.append(y[i])

    ###create a data frame###
    df=pd.DataFrame()
    df['texts']=txts
    df['text cleaned']=txts_cleaned
    df['folder name']=folder_array
    df['file name']=file_array
    ########################
    return (df)

df_train=load_and_clean_data(proj_dir+'20news-bydate-train')
df_test=load_and_clean_data(proj_dir+'20news-bydate-test')

"""# Creating Training,Validation and Testing Variables/Data

The labelled and grouped structure of this dataset necessitates that the training dataset be divided into training and validation sets, with the label distribution on both sets and the integrity of the groups being kept as near as feasible. To generate more representative training and validation sets, it is common practise to do data shuffles prior to model training. If the divided datasets aren't shuffled, the real distribution of the dataset won't be represented.

Our initial dataset must be shuffled in order to decrease variance and ensure that the model can generalise effectively to new, previously unknown datasets (testing data).

We utilise a RandomState seed to verify that the split is consistent. The StratifiedShuffleSplit module of sklearn, the shuffle() function in sklearn, and random.RandomState() method in NumPy were also examined. In our opinion, this appears to have solved the dataset's label stratification problem.
"""

#Shuffling Data for Splitting Randomly
seed=1337
rng = np.random.RandomState(seed)
rng.shuffle(df_train.texts.values.tolist())
rng = np.random.RandomState(seed)
rng.shuffle(np.unique(df_train['folder name'].values.tolist()))

#Extracted the Required Data.
X = df_train['texts']
y = df_train['folder name']
X_test=df_test['texts']
y_test=df_test['folder name']

#Splitting the shuffled Training data into Training and Validation set
sss = StratifiedShuffleSplit(n_splits=2, test_size=0.2, random_state=0)
sss.get_n_splits(X, y)
print(sss)
for train_index, val_index in sss.split(X, y):
  print("train_index:", train_index, "val_index:", val_index)
  X_train, X_val = X[train_index], X[val_index]
  Y_train, Y_val = y[train_index], y[val_index]

import numpy as np
import tensorflow as tf
from tensorflow import keras

"""# **Creating Text Vectorization from Training Dataset**

With the release of TF 2.1, several new features were added, including TextVectorization. This layer preprocesses raw text by normalizing/standardizing the text, tokenizing it, generating n-grams, and indexing the vocabulary. For additional information, see (https://www.tensorflow.org/api docs/python/tf/keras/layers/TextVectorization). 

This layer provides the fundamental capabilities for handling text in a Keras model. It converts a batch of strings (one example = one string) to either a list of token indices (1D tensor of integer token indices) or a dense representation (1D tensor of float values reflecting info about the example's tokens). 

This layer is responsible for processing natural language inputs. To be able to work with simple string inputs (categorical strings or pre-tokenized strings). The layer's vocabulary must be given at construction or acquired during adaptation (). After adaptation, it examines the dataset, determines the frequency of individual string values, and constructs a vocabulary from them. The following steps are involved in the processing:

Standardize each sentence (often by lowering the case and removing the punctuation).
Each sentence should be split into substrings.
Substrings should be recombined into tokens.
Tokens of the index (associate a unique int value with each token)
Transform each example into a vector of ints or a dense float vector using this index.

This section/layer of the project converts the characteristics of training data to integer sequences. We select a length of [100,300] for the sequence and a maximum token size of [20,000,40,000] for the max tokens.

Additionally, we tested the finished vectorizer to confirm that it functioned properly.
"""

#Creating vectorizer and adapting to the training data to be use for all data,restriced to 20,000 vocabulary
from tensorflow.keras.layers import TextVectorization
vectorizer = TextVectorization(max_tokens=30000, output_sequence_length=200)
text_ds = tf.data.Dataset.from_tensor_slices(X_train.values.tolist())
vectorizer.adapt(text_ds)

#testing vectorizer
vectorizer.get_vocabulary()[0:5]

#Creating word_index mapping to the vocabulary created from the vectorizer
voc = vectorizer.get_vocabulary()
word_index = dict(zip(voc, range(len(voc))))

#Testing word_index
test = ["cat", "sat", "mat"]
[word_index[w] for w in test]

"""### Get Embeddings-Using GLOVE word representation
GloVe is an unsupervised learning algorithm for obtaining vector representations for words

Word embedding is a learned representation for text where words that have the same meaning have a similar representation. There are different models used for word embedding tasks. the two most popular word embedding models, Word2Vec and Glove. GloVe is an unsupervised learning algorithm for obtaining vector representations for words, while word2vec is neural network model. Both are implemented in Python using the API provided by Gensim. Both do the same thing and give similar results. We decided on GloVe because of model simplicity in greating the embedding layer. More can be learned about the GloVe on Stanford’s (https://nlp.stanford.edu/projects/glove/)

This is created by Stanford University. Glove has pre-defined dense vectors for around every 6 billion words of English literature along with many other general use characters like comma, braces, and semicolons. 

We download all the vector type, but we only test the 100 and 200 dimension vector for our embedding layer. 200 seens to give a better result viz a vis other parameter configuration.
"""

!wget http://nlp.stanford.edu/data/glove.6B.zip
!unzip -q glove.6B.zip

"""# **Training the Embedding**

Word embeddings provide a method of utilising an efficient, dense representation in which words that are related in meaning have a similar encoding. It is important to note that you do not have to supply this encoding manually. An embedding is a dense vector of floating point values that has been constructed (the length of the vector is a parameter you specify). Instead of having to define the values for the embedding explicitly, they may be learnt through training (weights learned by the model during training, in the same way a model learns weights for a dense layer).

We use the 200 dimension vector and we got 400,000 vector words.
"""

embeddings_index = {}
f = open('/content/glove.6B.200d.txt',encoding='utf8')# source of the downloaded glove files-use the 200d.txt
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Total %s word vectors in Glove 6B 200d.' % len(embeddings_index))

"""In order to begin, we must first construct an embedding matrix using the embedding indexes of the GloVe word vector. An embedding matrix is a collection of all words and the embeddings that correspond to each of those words. A relationship representation problem for all of the words in the training dataset is addressed by the concept of an embedding matrix. We employ Glove embedding to solve this problem, and we determine the dimension and required number of tokens, which in our case is the length of the vectorized vocalbulary.

Furthermore, we attempt to compare and contrast the number of hits and missed words between Glove embedding and our training vocabulary. The outcome is excellent, with more than 70% of the targets being hit.
"""

num_tokens = len(voc) + 2
embedding_dim = 200
hits = 0
misses = 0
embedding_matrix = np.random.random((num_tokens, embedding_dim))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        hits += 1
    else:
        misses += 1
print("Converted %d words (%d misses)" % (hits, misses))

"""# **PreTrained Embedding**

The pre-trained GloVe word embeddings have already been downloaded and will be used. A zip file containing 400K words and their embeddings was downloaded for this example. These word embeddings have already been stored for you in the data directory and will be used here.

We used the embedding matrix, number of tokens[20000,30000], and embedding dimension[100,200] to develop our neural network's pretrained embedding layer. Due to the fact that we are already employing training embedding in our Neural network, we have set the training to false.
"""

embedding_layer = Embedding(
num_tokens,
embedding_dim,
embeddings_initializer=keras.initializers.Constant(embedding_matrix),
trainable=False,)
embedding_layer

"""## **Preparing Data input for Models**

Because neural networks operate on vectors, we turned all of the training, validation, and testing data into an array. This is critical because neural networks are designed to perform operations on vectors. The vectors are represented as multidimensional arrays in your code. Vectors are useful in deep learning for a variety of reasons, one of which is a particular operation called dot product. The dot product of two vectors informs you how similar they are in terms of direction and is scaled by the magnitude of the two vectors that are being used to compute it.
"""

class_names=y.unique()
label_names=list(class_names)
label_index=(np.unique(label_names, return_inverse=True)[1])
label_code=(np.unique(y, return_inverse=True)[1])

print(X_train.shape, X_val.shape)
print(Y_train.shape, Y_val.shape)

x_train = vectorizer(np.array([[s] for s in X_train.values.tolist()])).numpy()
x_val = vectorizer(np.array([[s] for s in X_val.values.tolist()])).numpy()
x_test = vectorizer(np.array([[s] for s in X_test.values.tolist()])).numpy()
type(x_test)

y_train=np.array(np.unique(Y_train, return_inverse=True)[1])
y_val=(np.unique(Y_val, return_inverse=True)[1])
y_test=(np.unique(y_test, return_inverse=True)[1])

"""# **1 dimensional CNN | Conv1D Maxpooling and GlobalMax Pooling Using PreTrained Embedding**

Feature mapping are made by convolutional layers of a convolutional neural network by applying learned filters to input data over and over again. It's easier for lower-level features to be learned at input levels nearer the input, while higher-level features (like forms and specific patterns) are learned at input levels deeper in the model. This is because convolutional layers in deep models are stacked.

Maximum pooling or max pooling is a pooling process that looks for the biggest value in each patch of each feature map. This process is called maximum pooling or max pooling.

Because average pooling would only show how common a feature is, down sampled or pooled data show the patch's most prominent feature instead, which is what you get with this method.

Another type of pooling, called global pooling, is used from time to time.

It's called "global pooling" when you don't down sample parts of the input feature data. Instead, the entire feature data is down sampled to one value. When setting the pool size to match the size of an input feature data, this is the same thing as setting the pool size to match.

Global pooling can be used to quickly summarise a feature in a data set. If you don't want to use a fully linked layer, you can also move from feature data to the model's predictions by using a linked layer.

In this Conv1D one dimensional convolutional neural network, we use Maxpooling and GlobalMax Pooling with size of five close to input and output dense layer respectively.

We also, use the pretrained embedding layer to train the model. Callbacl, earlystopping and modelcheckpoint are implemented to optimise the process.

Our choice of loss sparse_categorical_crossentropy", optimizer="rmsprop" are fully inline with the nature of our dataset. Various activition function-RELU and Softmax. In this model a dropout of 0.5 was applied at the end.

The testing accuracy 73.81% is highest, with 200 embedding dimension(D) and 30,000 vocabulary(Val) as again 100D and 20,000(Val)
"""

from tensorflow.keras import layers

int_sequences_input = keras.Input(shape=(None,), dtype="int64")
embedded_sequences = embedding_layer(int_sequences_input)
x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.MaxPooling1D(5)(x)
x = layers.Conv1D(128, 5, activation="relu")(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)
preds = layers.Dense(len(label_names), activation="softmax")(x)
model1 = keras.Model(int_sequences_input, preds)
model1.summary()

"""# **Graphical represention of Model1**"""

keras.utils.plot_model(model1, "multi_input_and_output_model.png", show_shapes=True)

"""# **Training Model1**"""

callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 10 epochs"
        patience=10,
        verbose=1,
    )
]

epochs = 50
batch_size=64
model1.compile(
    loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"]
)
cp=ModelCheckpoint('model_1.hdf5',monitor='val_acc',verbose=1,save_best_only=True)
history1=model1.fit(x_train, y_train, epochs=epochs, validation_data=(x_val, y_val),batch_size=batch_size,callbacks=callbacks)

"""# **Plotting the Accuracy and validation Loss for model1**"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history1.history["loss"])
ax1.plot(history1.history["val_loss"])
ax1.legend(["train", "val"], loc="upper right")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")

ax2.plot(history1.history["acc"])
ax2.plot(history1.history["val_acc"])
ax2.legend(["train", "val"], loc="upper right")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
plt.show()

"""# **Testing Model1 with testing dataset**"""

_, test_accuracy = model1.evaluate(x=x_test, y=y_test, verbose=0)
print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")

"""# **Predicting using unknow raw text-Model1**"""

string_input = keras.Input(shape=(1,), dtype="string")
x = vectorizer(string_input)
preds = model1(x)
end_to_end_model = keras.Model(string_input, preds)

probabilities = end_to_end_model.predict([['Update and Re-fit a Model Call Description. update will update and (by default) re-fit a model']])

class_names[np.argmax(probabilities[0])]

"""## **1 dimensional CNN | Conv1D-without using Pre-trained Embedding**

Similar to the above model this one dimensional convolution without the pretrained embedding,we use tensorflow pure embedding.With similar features(activation,loss, optimazer etc). The testing accuracy 73.71% is highest, with 200 embedding dimension(D) and 30,000 vocabulary(Val) as again 100D and 20,000(Val). Slightly less that the above model.
"""

import tensorflow as tf

from tensorflow.keras import layers

# A integer input for vocab indices.
inputs = tf.keras.Input(shape=(None,), dtype="int64")

# Next, we add a layer to map those vocab indices into a space of dimensionality
# 'embedding_dim'.
x = layers.Embedding(num_tokens, embedding_dim)(inputs)
x = layers.Dropout(0.5)(x)

# Conv1D + global max pooling
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.Conv1D(128, 7, padding="valid", activation="relu", strides=3)(x)
x = layers.GlobalMaxPooling1D()(x)

# We add a vanilla hidden layer:
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.5)(x)

# We project onto a multiclass unit output layer, and squash it with a softmax:
predictions = layers.Dense(len(label_names), activation="softmax", name="predictions")(x)

model2 = tf.keras.Model(inputs, predictions)

# Compile the model with multiclass crossentropy loss and an rmsprop optimizer.
model2.compile(loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["accuracy"])
print("Simple Neural Network")
model2.summary()

"""# **Graphical representation of Model2**

"""

keras.utils.plot_model(model2, "multi_input_and_output_model.png", show_shapes=True)

"""# **Training of Model2**"""

callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=10,
        verbose=1,
    )
]

epochs = 50
batch_size=64
# Fit the model using the train and test datasets.
cp=ModelCheckpoint('model_1.hdf5',monitor='val_acc',verbose=1,save_best_only=True)
history2=model2.fit(x_train, y_train,validation_data=(x_val, y_val), epochs=epochs,batch_size=batch_size,callbacks=callbacks)

"""# **Plotting Accuracy and validation loss of Model2**"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history2.history["loss"])
ax1.plot(history2.history["val_loss"])
ax1.legend(["train", "val"], loc="upper right")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")

ax2.plot(history2.history["accuracy"])
ax2.plot(history2.history["val_accuracy"])
ax2.legend(["train", "val"], loc="upper right")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
plt.show()

"""# **Testing Model2 with Test dataset**"""

_, test_accuracy = model2.evaluate(x=x_test, y=y_test, verbose=0)
print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")

"""# **Predicting unknown text with Model2**"""

string_input = keras.Input(shape=(1,), dtype="string")
x = vectorizer(string_input)
preds = model2(x)
end_to_end_model = keras.Model(string_input, preds)

probabilities = end_to_end_model.predict(
    [['As a sanity check, if the embedding matrix has been generated properly. In the above, when I saw the first five entries of\nthe training set, the first entry was\n']]
)

class_names[np.argmax(probabilities[0])]
#df_train.texts[500]

"""# **RNN-Bidirectional LSTM Model 50 Unit**

Recurrent neural networks are a sort of neural network in which past time steps' outputs are used as inputs in the current time step.
Recurrent neural networks with Long Short-Term Memory (LSTM) are one of the most intriguing kinds of deep learning at the time.

Unlike other recurrent neural networks, the network’s internal gates allow the model to be trained successfully using backpropagation through time, and avoid the vanishing gradients problem.

They've been utilised to show world-class performance in areas as diverse as language translation, automatic picture captioning, and text production.

LSTMs vary from multilayer Perceptrons and convolutional neural networks in that they are especially developed to solve sequence prediction issues. Bidirectional LSTMs are a kind of LSTM that may be used to increase model performance in sequence classification issues.

Bidirectional Recurrent Neural Networks (RNNs) have a simple concept.
It entails replicating the network's initial recurrent layer so that there are two layers side by side, then feeding the input sequence as is to the first layer and a reversed duplicate of the input sequence to the second.
On the input sequence, bidirectional LSTMs train two LSTMs instead of one. The first is based on the original input sequence, while the second is based on a reversed replica of the original input sequence.

In this model the bidrectional LSTM improve performance, with similar hyperparameter like the first model. We use the pretrained embedding and loss='sparse_categorical_crossentropy',optimizer='adam',50 units of LSTM neutron and softmax activation. 

The test result is 78.4% with optimise epoch of less than 20, with batch size of 30. Most importantly the number of units[50,500]. This result is the best we could get with up to 300units, the result decline after 300units. This is a very good result and Bidirectional LSTM is know to improve performance.
"""

sequence_input = Input(shape=(200,), dtype='int64')
embedded_sequences = embedding_layer(sequence_input)
l_lstm = Bidirectional(LSTM(units=300))(embedded_sequences)
preds = Dense(len(label_names), activation='softmax')(l_lstm)
model3 = Model(sequence_input, preds)
model3.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])

print("Bidirectional LSTM")
model3.summary()

"""# **Graphical representation of model3**"""

keras.utils.plot_model(model3, "multi_input_and_output_model.png", show_shapes=True)

"""# **Training Model3**"""

callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=10,
        verbose=1,
    )
]

epochs=30
batch_size=20
cp=ModelCheckpoint('model_1.hdf5',monitor='val_acc',verbose=1,save_best_only=True)
history3=model3.fit(x_train, y_train, validation_data=(x_val,y_val),epochs=epochs, batch_size=batch_size,callbacks=(callbacks,cp))

"""## **Plotting accuracy and loss of Model3**"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history3.history["loss"])
ax1.plot(history3.history["val_loss"])
ax1.legend(["train", "val"], loc="upper right")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")

ax2.plot(history3.history["acc"])
ax2.plot(history3.history["val_acc"])
ax2.legend(["train", "val"], loc="upper right")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
plt.show()

"""# **Testing Model3 on Testing data**"""

_, test_accuracy = model3.evaluate(x=x_test, y=y_test, verbose=0)
print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")

from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

def create_model():
  sequence_input = Input(shape=(200,), dtype='int64')
  embedded_sequences = embedding_layer(sequence_input)
  l_lstm = Bidirectional(LSTM(units=300))(embedded_sequences)
  preds = Dense(len(label_names), activation='softmax')(l_lstm)
  model3 = Model(sequence_input, preds)
  model3.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['acc'])
  return model3

"""# **Predicting with Model3**"""

string_input = keras.Input(shape=(1,), dtype="string")
x = vectorizer(string_input)
preds = model3(x)
end_to_end_model = keras.Model(string_input, preds)

probabilities = end_to_end_model.predict(
    [['Jon playes cricket His favourite player is MS DhoniSometimes he loves to play football']]
)

class_names[np.argmax(probabilities[0])]

"""# **1 dimensional CNN MODEL- Developed as a function**"""

def ConvNet1(embeddings, num_tokens, embedding_dim, label_code):
    
    embedding_layer = Embedding(
        num_tokens,
        embedding_dim,
        embeddings_initializer=keras.initializers.Constant(embedding_matrix),
        trainable=False,)
    
    sequence_input = Input(shape=(None,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)
    x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(128, 5, activation="relu")(x)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(128, 5, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    preds = Dense(len(label_names), activation='softmax')(x)
    model = keras.Model(sequence_input, preds)
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer="adam",
                  metrics=['acc'])
    model.summary()
    return model

model4 = ConvNet1(embedding_matrix, num_tokens,embedding_dim,label_code)

"""# **Graph of Model4**"""

keras.utils.plot_model(model4, "multi_input_and_output_model.png", show_shapes=True)

"""# **Training of Model4**"""

callbacks = [
    keras.callbacks.EarlyStopping(
        # Stop training when `val_loss` is no longer improving
        monitor="val_loss",
        # "no longer improving" being defined as "no better than 1e-2 less"
        min_delta=1e-2,
        # "no longer improving" being further defined as "for at least 2 epochs"
        patience=10,
        verbose=1,
    )
]

epochs=50
batch_size=64
cp=ModelCheckpoint('model_1.hdf5',monitor='val_acc',verbose=1,save_best_only=True)

model4 = ConvNet1(embedding_matrix, num_tokens, embedding_dim,label_code)
history4 = model4.fit(x_train, y_train, epochs=epochs, shuffle=True,validation_data=(x_val,y_val), batch_size=batch_size,callbacks=(callbacks,cp))

"""## **Plotting accuracy and loss of Model4**"""

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(history4.history["loss"])
ax1.plot(history4.history["val_loss"])
ax1.legend(["train", "val"], loc="upper right")
ax1.set_xlabel("Epochs")
ax1.set_ylabel("Loss")

ax2.plot(history4.history["acc"])
ax2.plot(history4.history["val_acc"])
ax2.legend(["train", "val"], loc="upper right")
ax2.set_xlabel("Epochs")
ax2.set_ylabel("Accuracy")
plt.show()

"""## **Testing Model4**"""

_, test_accuracy = model4.evaluate(x=x_test, y=y_test, verbose=0)
print(f"Test accuracy: {round(test_accuracy * 100, 2)}%")

"""# **Predicting with Model4**"""

string_input = keras.Input(shape=(1,), dtype="string")
x = vectorizer(string_input)
preds = model4(x)
end_to_end_model = keras.Model(string_input, preds)

probabilities = end_to_end_model.predict(
    [["Jon playes cricket His favourite player is MS Dhoni Sometimes he loves to play football"]]
)

class_names[np.argmax(probabilities[0])]




#######################################################################

NoteBook 2_0_Analysis_of_20_NewsGroups_Data_Submission.ipynb

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
## License:
The 20NewsGroup data is open source
All code owned by Cardiff University


### END ###
