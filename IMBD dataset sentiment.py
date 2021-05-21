{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "885b14b1",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import nltk\n",
    "from nltk import word_tokenize\n",
    "from bs4 import BeautifulSoup\n",
    "from nltk.stem.porter import PorterStemmer\n",
    "import re\n",
    "from textblob import TextBlob\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "from sklearn.preprocessing import LabelEncoder\n",
    "from sklearn.linear_model import LogisticRegression , SGDClassifier\n",
    "from sklearn.metrics import accuracy_score , classification_report , confusion_matrix\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.naive_bayes import MultinomialNB"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "bb54bffb",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"E:\\dataset\\imbd reviews\\IMDB Dataset.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "6390af48",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review</th>\n",
       "      <th>sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>One of the other reviewers has mentioned that ...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>A wonderful little production. &lt;br /&gt;&lt;br /&gt;The...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>I thought this was a wonderful way to spend ti...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Basically there's a family where a little boy ...</td>\n",
       "      <td>negative</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Petter Mattei's \"Love in the Time of Money\" is...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              review sentiment\n",
       "0  One of the other reviewers has mentioned that ...  positive\n",
       "1  A wonderful little production. <br /><br />The...  positive\n",
       "2  I thought this was a wonderful way to spend ti...  positive\n",
       "3  Basically there's a family where a little boy ...  negative\n",
       "4  Petter Mattei's \"Love in the Time of Money\" is...  positive"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "eadfde53",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "review       0\n",
       "sentiment    0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "a58035f9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review</th>\n",
       "      <th>sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>50000</td>\n",
       "      <td>50000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unique</th>\n",
       "      <td>49582</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>top</th>\n",
       "      <td>Loved today's show!!! It was a variety and not...</td>\n",
       "      <td>negative</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>freq</th>\n",
       "      <td>5</td>\n",
       "      <td>25000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                                   review sentiment\n",
       "count                                               50000     50000\n",
       "unique                                              49582         2\n",
       "top     Loved today's show!!! It was a variety and not...  negative\n",
       "freq                                                    5     25000"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "f9ba1447",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "negative    25000\n",
       "positive    25000\n",
       "Name: sentiment, dtype: int64"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['sentiment'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "d0b679b2",
   "metadata": {},
   "outputs": [],
   "source": [
    "stopwords = nltk.corpus.stopwords.words('english')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "6c77ec3e",
   "metadata": {},
   "outputs": [],
   "source": [
    "## removing noise ##"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "82913bfa",
   "metadata": {},
   "outputs": [],
   "source": [
    "def remove_noise(text):\n",
    "    soup = BeautifulSoup(text , \"html.parser\")\n",
    "    text = soup.get_text()\n",
    "    text = text.strip(\"[]\")\n",
    "    return text\n",
    "    \n",
    "df['review'] = df['review'].apply(remove_noise)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "79b4be23",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review</th>\n",
       "      <th>sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>One of the other reviewers has mentioned that ...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>A wonderful little production. The filming tec...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>I thought this was a wonderful way to spend ti...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Basically there's a family where a little boy ...</td>\n",
       "      <td>negative</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Petter Mattei's \"Love in the Time of Money\" is...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              review sentiment\n",
       "0  One of the other reviewers has mentioned that ...  positive\n",
       "1  A wonderful little production. The filming tec...  positive\n",
       "2  I thought this was a wonderful way to spend ti...  positive\n",
       "3  Basically there's a family where a little boy ...  negative\n",
       "4  Petter Mattei's \"Love in the Time of Money\" is...  positive"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "a8f30a09",
   "metadata": {},
   "outputs": [],
   "source": [
    "## definition a function for removing special character ##"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "789c9dbe",
   "metadata": {},
   "outputs": [],
   "source": [
    "def removing_special_character(text,remove_digits = True):\n",
    "    pattern = r'[^a-zA-Z0-9\\s]'\n",
    "    text = re.sub(pattern, '', text)\n",
    "    return text\n",
    "\n",
    "df['review'] = df['review'].apply(removing_special_character)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "279ba41e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review</th>\n",
       "      <th>sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>One of the other reviewers has mentioned that ...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>A wonderful little production The filming tech...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>I thought this was a wonderful way to spend ti...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Basically theres a family where a little boy J...</td>\n",
       "      <td>negative</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Petter Matteis Love in the Time of Money is a ...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              review sentiment\n",
       "0  One of the other reviewers has mentioned that ...  positive\n",
       "1  A wonderful little production The filming tech...  positive\n",
       "2  I thought this was a wonderful way to spend ti...  positive\n",
       "3  Basically theres a family where a little boy J...  negative\n",
       "4  Petter Matteis Love in the Time of Money is a ...  positive"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "e326ad05",
   "metadata": {},
   "outputs": [],
   "source": [
    "## correcting spells ##"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "d6947680",
   "metadata": {},
   "outputs": [],
   "source": [
    "#df['review'] = df['review'].apply(lambda review: TextBlob(review).correct())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "dc32e983",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review</th>\n",
       "      <th>sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>One of the other reviewers has mentioned that ...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>A wonderful little production The filming tech...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>I thought this was a wonderful way to spend ti...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Basically theres a family where a little boy J...</td>\n",
       "      <td>negative</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Petter Matteis Love in the Time of Money is a ...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              review sentiment\n",
       "0  One of the other reviewers has mentioned that ...  positive\n",
       "1  A wonderful little production The filming tech...  positive\n",
       "2  I thought this was a wonderful way to spend ti...  positive\n",
       "3  Basically theres a family where a little boy J...  negative\n",
       "4  Petter Matteis Love in the Time of Money is a ...  positive"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "5a2f455b",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Text steamming ##"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "0aac120c",
   "metadata": {},
   "outputs": [],
   "source": [
    "def clean_text(text):\n",
    "    ps = nltk.porter.PorterStemmer()\n",
    "    text = text.lower()\n",
    "    text = ' '.join([word for word in text.split() if word not in stopwords])    ## Text without stopwords\n",
    "    text = ' '.join([ps.stem(word) for word in text.split()])\n",
    "    return text\n",
    "\n",
    "df['review'] = df['review'].apply(clean_text)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "acbd5432",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>review</th>\n",
       "      <th>sentiment</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>one review mention watch 1 oz episod youll hoo...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>wonder littl product film techniqu unassum old...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>thought wonder way spend time hot summer weeke...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>basic there famili littl boy jake think there ...</td>\n",
       "      <td>negative</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>petter mattei love time money visual stun film...</td>\n",
       "      <td>positive</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                              review sentiment\n",
       "0  one review mention watch 1 oz episod youll hoo...  positive\n",
       "1  wonder littl product film techniqu unassum old...  positive\n",
       "2  thought wonder way spend time hot summer weeke...  positive\n",
       "3  basic there famili littl boy jake think there ...  negative\n",
       "4  petter mattei love time money visual stun film...  positive"
      ]
     },
     "execution_count": 22,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "da7d9992",
   "metadata": {},
   "outputs": [],
   "source": [
    "df_train = df[:35000]\n",
    "df_test = df[:15000]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "f6176b4b",
   "metadata": {},
   "outputs": [],
   "source": [
    "tfidf_vectorizer = TfidfVectorizer(use_idf=True, ngram_range=(1,3))\n",
    "\n",
    "tfidf_train = tfidf_vectorizer.fit_transform(df_train['review'])\n",
    "tfidf_test = tfidf_vectorizer.transform(df_test['review']) #-----------------------------> "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "21ea51a4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "tfidf_train shape (35000, 6096399)\n",
      "tfidf_test shape (15000, 6096399)\n"
     ]
    }
   ],
   "source": [
    "print('tfidf_train shape' ,tfidf_train.shape)\n",
    "print('tfidf_test shape' ,tfidf_test.shape)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "743262b5",
   "metadata": {},
   "outputs": [],
   "source": [
    "le = LabelEncoder()\n",
    "sentiment_data = le.fit_transform(df_train['sentiment'])             #TRAIN SENTIMENT\n",
    "sentiment_test_data = le.fit_transform(df_test['sentiment'])         #TEST SENTIMENT\n",
    "\n",
    "#print(sentiment_test_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "a9e929fa",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Applying logistic regression ##"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "a7eac8a8",
   "metadata": {},
   "outputs": [],
   "source": [
    "LR = LogisticRegression(penalty='l2' , C=1 , max_iter= 300 , random_state = 42)\n",
    "LR_tfidf = LR.fit(tfidf_train , sentiment_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "11337811",
   "metadata": {},
   "outputs": [],
   "source": [
    "## Checking performence on test data ##"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "cc7bc328",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 1 1 ... 0 0 1]\n"
     ]
    }
   ],
   "source": [
    "LR_tfidf_pred = LR.predict(tfidf_test)\n",
    "print(LR_tfidf_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "094e7c7d",
   "metadata": {},
   "outputs": [],
   "source": [
    "################################# Checking accuracy #####################################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "b1dbbb6f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.9668666666666667"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "LR_tfidf_accuracy = accuracy_score(sentiment_test_data , LR_tfidf_pred)\n",
    "LR_tfidf_accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "aeb789c9",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "88580806",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 33,
   "id": "bee5654b",
   "metadata": {},
   "outputs": [],
   "source": [
    "############################3#### Classificatopn report ####################################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "7a8bef72",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "    positive       0.97      0.96      0.97      7609\n",
      "    negative       0.96      0.97      0.97      7391\n",
      "\n",
      "    accuracy                           0.97     15000\n",
      "   macro avg       0.97      0.97      0.97     15000\n",
      "weighted avg       0.97      0.97      0.97     15000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "LR_tfidf_report = classification_report(sentiment_test_data, LR_tfidf_pred , target_names=['positive' , 'negative'])\n",
    "print(LR_tfidf_report)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "0878c877",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[7317  292]\n",
      " [ 205 7186]]\n"
     ]
    }
   ],
   "source": [
    "############################## Confusion matrix ############################################\n",
    "LR_tfidf_matrix = confusion_matrix(sentiment_test_data , LR_tfidf_pred)\n",
    "print(LR_tfidf_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "f7f33e37",
   "metadata": {},
   "outputs": [],
   "source": [
    "########################## Testing on svm model ###################################333"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "928b22a1",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "SGDClassifier(max_iter=500, random_state=42)\n"
     ]
    }
   ],
   "source": [
    "svm=SGDClassifier(loss='hinge',max_iter=500,random_state=42)\n",
    "\n",
    "svm_tfidf = svm.fit(tfidf_train , sentiment_data)\n",
    "print(svm_tfidf)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "05ebc503",
   "metadata": {},
   "outputs": [],
   "source": [
    "#################### svm predicting on test data ####################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "947cd9e5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[1 1 1 ... 0 0 1]\n"
     ]
    }
   ],
   "source": [
    "svm_tfidf_pred = svm.predict(tfidf_test)\n",
    "print(svm_tfidf_pred)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "dd7c10a4",
   "metadata": {},
   "outputs": [],
   "source": [
    "################### svm accuracy ############################"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "18eba70d",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "0.9552\n"
     ]
    }
   ],
   "source": [
    "svm_accuracy = accuracy_score(sentiment_test_data , svm_tfidf_pred)\n",
    "print(svm_accuracy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "837dce71",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 42,
   "id": "bc543ae5",
   "metadata": {},
   "outputs": [],
   "source": [
    "################ svm classification report #######################3"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "18d16350",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.97      0.94      0.96      7609\n",
      "           1       0.94      0.97      0.96      7391\n",
      "\n",
      "    accuracy                           0.96     15000\n",
      "   macro avg       0.96      0.96      0.96     15000\n",
      "weighted avg       0.96      0.96      0.96     15000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "svm_report = classification_report(sentiment_test_data,svm_tfidf_pred)\n",
    "print(svm_report)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "8b95b447",
   "metadata": {},
   "outputs": [],
   "source": [
    "##################$$$$$$$$$$$ confusion matrix ###############################33"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "25fb8c64",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[7145  464]\n",
      " [ 208 7183]]\n"
     ]
    }
   ],
   "source": [
    "svm_matrix = confusion_matrix(sentiment_test_data , svm_tfidf_pred)\n",
    "print(svm_matrix)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "e9c72d59",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 46,
   "id": "214866c7",
   "metadata": {},
   "outputs": [],
   "source": [
    "#### USING MULTINOMIAL  NAVIE BIAS #################################33"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "e08aeefd",
   "metadata": {},
   "outputs": [],
   "source": [
    "nb = MultinomialNB()\n",
    "nb_tfidf = nb.fit(tfidf_train , sentiment_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "id": "6871ea49",
   "metadata": {},
   "outputs": [],
   "source": [
    "nb_tfidf_predict = nb.predict(tfidf_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 49,
   "id": "83de4d99",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.997"
      ]
     },
     "execution_count": 49,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "nb_score = accuracy_score(sentiment_test_data , nb_tfidf_predict)           ##### whoooooooooooooo look at the score ########\n",
    "nb_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 50,
   "id": "34a9f9be",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "    Positive       1.00      1.00      1.00      7609\n",
      "    Negative       1.00      1.00      1.00      7391\n",
      "\n",
      "    accuracy                           1.00     15000\n",
      "   macro avg       1.00      1.00      1.00     15000\n",
      "weighted avg       1.00      1.00      1.00     15000\n",
      "\n"
     ]
    }
   ],
   "source": [
    "nb_report = classification_report(sentiment_test_data , nb_tfidf_predict,target_names=['Positive','Negative'])\n",
    "print(nb_report)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "id": "8c946560",
   "metadata": {},
   "outputs": [],
   "source": [
    "#######################3 confusion matrix #############################"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "d7ed565f",
   "metadata": {},
   "source": [
    "nb_report = confusion_matrix(sentiment_test_data,nb_tfidf_predict)\n",
    "print(nb_report)"
   ]
  },
  {
   "cell_type": "raw",
   "id": "29172943",
   "metadata": {},
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
