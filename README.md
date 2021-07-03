# textMining

Q1.1 

The first section of the program returns the possible sentiments by outputting
all the unique attributes in the Sentiments column. 

The second most popular sentiments are programmed by sorting by the sentiments and their value counts which show
the the sorted frequency for the sentiments and taking the [1] value which is the second highest. 

To find the greatest number of extremely positive tweets, a new dataframe was created which only takes the 
'Extremely Positive' sentiments from the sentiment column and keeps track of the 'TweetAt' column which is the date. 
From here, the value counts are taking to get the date with the most number of extremely positive tweets. 

To convert the tweets into lower case and replacing the non-alphabetical characters with whitespaces, regex was used
and every non-alphabetical character was replaced with a white splace and changed to lower case. 

Q1.2 

To tokenize the tweets, the column of tweets that were converted in the last section of 1.1 were split by single
whitespaces. This allowed for every word in the tweets to be separated and every tweet shown as a list.

Raw python string functions were used to count the total number of words

To find the unique words within the tweets, the words were concatenated into one large list and then converted into
a set because finding the length of the sets will only output unique values in the list. 

The nltk library was used to find the 10 most frequent words, the FreqDist was used and set to 10 to output the 10
most frequently occuring words in the tweets column.

To remove stop words, instead of the nltk library, the sklearn library contained a list of stop words which were used
to remove from the tweets using regex and words with less than 3 characters were also removed using regex.

The same method was then used to count the total number of words in the new dataframe and the most frequent words in
the modified corpus.

Q1.3 

To create the linegraph with word frequencies, the count vectorizer from the sklearn library was used to separate the
words and add them to a list and the count was used to find the frequency of the words which was then sorted in 
ascending order.

Q1.4 

The count vectorizer was also used to create a multinomial naive bayes classifier for the Coronavirus NLP Train 
dataset. The vectorized tweets from the original dataset were fitted with the multinomial naive bayes classifier
with the sentiment and the prediction was run. The accuracy was then calculated by counting the number of times the 
prediction correctly met the target value. The error rate was taken as 1 - the accuracy value outputted. 

