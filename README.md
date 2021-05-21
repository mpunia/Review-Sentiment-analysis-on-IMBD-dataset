# Review-Sentiment-analysis-on-IMBD-dataset

### In the dataset we have 50000 have rows and two columns for review and sentiment, review contains viewer reviews and sentiment contains if it is positibe and negative.
We gave effort in cleaning the data i.e review column, and make it efficient by removing link stopword and applying stem. Then on the sentiment column we applied encoding
0 for 'encoding' 1 for 'positive' . Then we applied Tfidf-vectorizer to vectorize review columns so that we can feed it our machine learning model. After we a applied 3 algorithm.
The heighest accuracy we got was 99.5 with the Multinomial Naive Bayes algorithm.  
