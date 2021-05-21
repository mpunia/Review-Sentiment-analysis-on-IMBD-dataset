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


# In[5]:


df = pd.read_csv("E:\dataset\imbd reviews\IMDB Dataset.csv")


# In[6]:


df.head()


# In[7]:


df.isnull().sum()


# In[8]:


df.describe()


# In[9]:


df['sentiment'].value_counts()


# In[10]:


stopwords = nltk.corpus.stopwords.words('english')


# In[11]:


## removing noise ##


# In[12]:


def remove_noise(text):
    soup = BeautifulSoup(text , "html.parser")
    text = soup.get_text()
    text = text.strip("[]")
    return text
    
df['review'] = df['review'].apply(remove_noise)


# In[13]:


df.head()


# In[14]:


## definition a function for removing special character ##


# In[15]:


def removing_special_character(text,remove_digits = True):
    pattern = r'[^a-zA-Z0-9\s]'
    text = re.sub(pattern, '', text)
    return text

df['review'] = df['review'].apply(removing_special_character)


# In[16]:


df.head()


# In[17]:


## correcting spells ##


# In[18]:


#df['review'] = df['review'].apply(lambda review: TextBlob(review).correct())


# In[19]:


df.head()


# In[20]:


## Text steamming ##


# In[21]:


def clean_text(text):
    ps = nltk.porter.PorterStemmer()
    text = text.lower()
    text = ' '.join([word for word in text.split() if word not in stopwords])    ## Text without stopwords
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

df['review'] = df['review'].apply(clean_text)


# In[22]:


df.head()


# In[23]:


df_train = df[:35000]
df_test = df[:15000]


# In[24]:


tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,3))

tfidf_train = tfidf_vectorizer.fit_transform(df_train['review'])
tfidf_test = tfidf_vectorizer.transform(df_test['review']) #-----------------------------> 


# In[25]:


print('tfidf_train shape' ,tfidf_train.shape)
print('tfidf_test shape' ,tfidf_test.shape)


# In[26]:


le = LabelEncoder()
sentiment_data = le.fit_transform(df_train['sentiment'])             #TRAIN SENTIMENT
sentiment_test_data = le.fit_transform(df_test['sentiment'])         #TEST SENTIMENT

#print(sentiment_test_data)


# In[27]:


## Applying logistic regression ##


# In[28]:


LR = LogisticRegression(penalty='l2' , C=1 , max_iter= 300 , random_state = 42)
LR_tfidf = LR.fit(tfidf_train , sentiment_data)


# In[29]:


## Checking performence on test data ##


# In[30]:


LR_tfidf_pred = LR.predict(tfidf_test)
print(LR_tfidf_pred)


# In[31]:


################################# Checking accuracy #####################################


# In[32]:


LR_tfidf_accuracy = accuracy_score(sentiment_test_data , LR_tfidf_pred)
LR_tfidf_accuracy


# In[ ]:





# In[ ]:





# In[33]:


############################3#### Classificatopn report ####################################


# In[34]:


LR_tfidf_report = classification_report(sentiment_test_data, LR_tfidf_pred , target_names=['positive' , 'negative'])
print(LR_tfidf_report)


# In[35]:


############################## Confusion matrix ############################################
LR_tfidf_matrix = confusion_matrix(sentiment_test_data , LR_tfidf_pred)
print(LR_tfidf_matrix)


# In[36]:


########################## Testing on svm model ###################################333


# In[37]:


svm=SGDClassifier(loss='hinge',max_iter=500,random_state=42)

svm_tfidf = svm.fit(tfidf_train , sentiment_data)
print(svm_tfidf)


# In[38]:


#################### svm predicting on test data ####################


# In[39]:


svm_tfidf_pred = svm.predict(tfidf_test)
print(svm_tfidf_pred)


# In[40]:


################### svm accuracy ############################


# In[41]:


svm_accuracy = accuracy_score(sentiment_test_data , svm_tfidf_pred)
print(svm_accuracy)


# In[ ]:





# In[42]:


################ svm classification report #######################3


# In[43]:


svm_report = classification_report(sentiment_test_data,svm_tfidf_pred)
print(svm_report)


# In[44]:


##################$$$$$$$$$$$ confusion matrix ###############################33


# In[45]:


svm_matrix = confusion_matrix(sentiment_test_data , svm_tfidf_pred)
print(svm_matrix)


# In[ ]:





# In[46]:


#### USING MULTINOMIAL  NAVIE BIAS #################################33


# In[47]:


nb = MultinomialNB()
nb_tfidf = nb.fit(tfidf_train , sentiment_data)


# In[48]:


nb_tfidf_predict = nb.predict(tfidf_test)


# In[49]:


nb_score = accuracy_score(sentiment_test_data , nb_tfidf_predict)           ##### whoooooooooooooo look at the score ########
nb_score


# In[50]:


nb_report = classification_report(sentiment_test_data , nb_tfidf_predict,target_names=['Positive','Negative'])
print(nb_report)


# In[51]:


#######################3 confusion matrix #############################


# nb_report = confusion_matrix(sentiment_test_data,nb_tfidf_predict)
# print(nb_report)
