#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import nltk
from nltk import word_tokenize
from bs4 import BeautifulSoup
from nltk.stem.porter import PorterStemmer
import re
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression , SGDClassifier
from sklearn.metrics import accuracy_score , classification_report , confusion_matrix
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("E:\dataset\imbd reviews\IMDB Dataset.csv")
df.head()

df.isnull().sum()


# In[8]:


df.describe()
df['sentiment'].value_counts()
stopwords = nltk.corpus.stopwords.words('english')

## removing noise ##

def remove_noise(text):
    soup = BeautifulSoup(text , "html.parser")
    text = soup.get_text()
    text = text.strip("[]")
    return text
    
df['review'] = df['review'].apply(remove_noise)

df.head()

## definition a function for removing special character ##

def removing_special_character(text,remove_digits = True):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, '', text)
    return text

df['review'] = df['review'].apply(removing_special_character)


#df['review'] = df['review'].apply(lambda review: TextBlob(review).correct())

df.head()

## Text steamming ##

def clean_text(text):
    ps = nltk.porter.PorterStemmer()
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords])    ## Text without stopwords
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

df['review'] = df['review'].apply(clean_text)

df_train = df[:35000]
df_test = df[:15000]

tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,3))

tfidf_train = tfidf_vectorizer.fit_transform(df_train['review'])
tfidf_test = tfidf_vectorizer.transform(df_test['review']) #-----------------------------> vectorize test data review

print('tfidf_train shape' ,tfidf_train.shape)
print('tfidf_test shape' ,tfidf_test.shape)

le = LabelEncoder()
sentiment_data = le.fit_transform(df_train['sentiment'])             #TRAIN SENTIMENT
sentiment_test_data = le.fit_transform(df_test['sentiment'])         #TEST SENTIMENT

#print(sentiment_test_data)

## Applying logistic regression ##

LR = LogisticRegression(penalty='l2' , C=1 , max_iter= 300 , random_state = 42)
LR_tfidf = LR.fit(tfidf_train , sentiment_data)

## Checking performence on test data ##

LR_tfidf_pred = LR.predict(tfidf_test)
print(LR_tfidf_pred)

################################# Checking accuracy #####################################


LR_tfidf_accuracy = accuracy_score(sentiment_test_data , LR_tfidf_pred)
LR_tfidf_accuracy


############################3#### Classificatopn report ####################################

LR_tfidf_report = classification_report(sentiment_test_data, LR_tfidf_pred , target_names=['positive' , 'negative'])
print(LR_tfidf_report)


############################## Confusion matrix ############################################
LR_tfidf_matrix = confusion_matrix(sentiment_test_data , LR_tfidf_pred)
print(LR_tfidf_matrix)

########################## Testing on svm model ###################################333

svm=SGDClassifier(loss='hinge',max_iter=500,random_state=42)

svm_tfidf = svm.fit(tfidf_train , sentiment_data)
print(svm_tfidf)


#################### svm predicting on test data ####################

svm_tfidf_pred = svm.predict(tfidf_test)
print(svm_tfidf_pred)


################### svm accuracy ############################


svm_accuracy = accuracy_score(sentiment_test_data , svm_tfidf_pred)
print(svm_accuracy)

################ svm classification report #######################3


svm_report = classification_report(sentiment_test_data,svm_tfidf_pred)
print(svm_report)

##################$$$$$$$$$$$ confusion matrix ###############################33


svm_matrix = confusion_matrix(sentiment_test_data , svm_tfidf_pred)
print(svm_matrix)


#### USING MULTINOMIAL  NAVIE BIAS #################################33

nb = MultinomialNB()
nb_tfidf = nb.fit(tfidf_train , sentiment_data)


nb_tfidf_predict = nb.predict(tfidf_test)

nb_score = accuracy_score(sentiment_test_data , nb_tfidf_predict)           ##### whoooooooooooooo look at the score ########
nb_score


nb_report = classification_report(sentiment_test_data , nb_tfidf_predict,target_names=['Positive','Negative'])
print(nb_report)


#######################3 confusion matrix #############################


nb_report = confusion_matrix(sentiment_test_data,nb_tfidf_predict)
print(nb_report)
