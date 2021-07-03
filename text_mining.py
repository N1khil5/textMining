import pandas as pd
import sklearn
import numpy as np
import nltk
import re
from nltk.probability import FreqDist
import matplotlib.pyplot as plt

df = pd.read_csv('Corona_NLP_train.csv', encoding='latin-1')

# Q1 Possible Sentiments
#print("Possible Sentiments = ", df['Sentiment'].unique())

# Q1 Second Most Popular Sentiment in the Tweets
#print("Second Largest = ", df['Sentiment'].value_counts().index.tolist()[1])

# Q1 Date with the greatest number of extremely positive tweets

exPosTweets = df.loc[df['Sentiment'] == 'Extremely Positive', ['TweetAt','Sentiment']]
exPosTweetsDate = exPosTweets.value_counts().index.tolist()[0]
exPosTweetNumber = exPosTweets.value_counts()[0]
#print("Date with most Extremely Positive tweets = ", exPosTweetsDate," , ", exPosTweetNumber)

# Q1 Convert messages to lower case

newDf = df['OriginalTweet'].str.lower()

# Q1 Replace non-alphabetical characters with whitespaces
newDf['RegexTweet'] = newDf.replace('[^a-zA-Z]+', " ", regex=True)
newDf['RegexTweet'] = newDf['RegexTweet'].replace(r'^\s',"", regex=True)
#print(newDf['RegexTweet'])

# Q2 Tokenize the tweets
newDf['RegexTweet'].str.split(" ").values

# Q2 Counting the total number of words (including repetitions)
count = newDf['RegexTweet'].str.split().str.len().sum()
#print("Count = ", count)

# Q2 Counting the number of all distinct words
distinctWords = newDf['RegexTweet'].str.split()
dw = np.concatenate(distinctWords)
uniqueWords = len(set(dw))
#print("Distinct Words = ", uniqueWords)

# Q2 Showing the 10 most frequent words in the corpus
most_frequent = FreqDist(dw)
#print(most_frequent.most_common(10))

# Q2 Removing stop words and words with <= 2 characters
stopWords = sklearn.feature_extraction.text.ENGLISH_STOP_WORDS

# Removing words with <= 2 characters
newDf['RegexTweet1'] = newDf['RegexTweet'].replace(r'\W*\b\w{1,2}\b',"", regex=True)
newDf['RegexTweet'] = newDf['RegexTweet1'].replace(r'^\s',"", regex=True)

# Removing STOP words
newDf['RegexTweet'] = newDf['RegexTweet'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stopWords)]))
modifiedCorpus = newDf['RegexTweet']
#print(modifiedCorpus)

# Q2 Number of all words after removal of stop words and <= characters
count = newDf['RegexTweet'].str.split().str.len().sum()
#print("Count = ", count)

# Q2 10 Most frequent words in the modified corpus
distinctWordsMod = newDf['RegexTweet'].str.split()
dwMod = np.concatenate(distinctWordsMod)
most_frequent = FreqDist(dwMod)
#print(most_frequent.most_common(10))

# Q3 Histogram with word frequencies
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
words = modifiedCorpus.to_list()
count = vectorizer.fit_transform(words)

freq = np.asarray(count.sum(axis=0))
freq = sorted(np.concatenate(freq))

#plt.plot(freq.index,freq)
plt.show()
#print(freq)

# Q4 Multinomial Naive Bayes Classifier for the Coronavirus Tweets NLP data set using scikit-learn
from sklearn.naive_bayes import MultinomialNB
import sklearn.metrics as metrics
x = df['OriginalTweet']
y = df['Sentiment']
x = vectorizer.fit_transform(x)
clf = MultinomialNB()
clf.fit(x, y)

prediction = clf.predict(x)

accuracy = metrics.accuracy_score(y, prediction)
error_rate = (1-accuracy) * 100
print(accuracy)
print(error_rate, "%")