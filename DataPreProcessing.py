from nltk.corpus import stopwords 
from string import punctuation 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer 
import pandas as pd
import numpy as np
import re


### Remove links, short words and symbols
def CleaningReviews(data):
    
    corpus = []
    stop_word = set(stopwords.words('english') + list(punctuation))
    # symbols = ['!','@','#','$','%','^','&','*','(',')','-','_','+','=',',','<','>','?','/','.',';',':','[',']','{','}',"|",'--','»']
    # stop_word.update(symbols)
    # print(stopwords.words())
    for review in data:
        ### removing links 
        review = re.sub(r'''(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’]))''', '', review, flags=re.MULTILINE)
        ### break review into tokens               
        text_tokens = word_tokenize(review)
        # print(text_tokens)
        ### Renove short words and symbols
        tokens_without_sw = [word for word in text_tokens if not word in stop_word]
        # print(tokens_without_sw)
        # print('\n')
        ### Combine tokens together and save them into a corpus(list)
        filtered_sentence = (" ").join(tokens_without_sw)
        corpus.append(filtered_sentence)
        # print(corpus)
    return corpus 


### map words in coupus to their base word e.g. 'reading' will become 'read'
def ReviewLemmatization(data):
    lemmatizer = WordNetLemmatizer() 
    corpus = []
    for review in data:
        lammatizedWords = []
        text_tokens = word_tokenize(review)
        for i in range(0,len(text_tokens)):
            # Lemmatize the words
            lemmatized_words = lemmatizer.lemmatize(text_tokens[i])
            lammatizedWords.append(lemmatized_words)
            
        # print(lammatizedWords)
        # print('\n')
        Lammetized_words = (" ").join(lammatizedWords)
        corpus.append(Lammetized_words)     
    
    return corpus                     
            




data = pd.read_csv("Uber_Data_review.csv", encoding = "ISO-8859-1")
uberData = []

reviews =  data.ride_review
for i in range(0,len(reviews)):
    ### remove the numbers from reviews and save them into uberData
    result = re.sub(r'\d+', '',reviews[i]) 
    uberData.append(result)

### lowercase all the words in list   
uberData = [review.lower() for review in uberData] 
uberData = CleaningReviews(uberData)
uberData = ReviewLemmatization(uberData)


data['CleanRide_review'] = uberData
data['sentiment'] = np.where(data['ride_rating']>3,1,0)


column_titles = ['CleanRide_review','ride_rating','sentiment','ride_review']
data_reordered = data.reindex(columns = column_titles)
data_reordered = data_reordered.drop('ride_review', axis = 1) 
### create a new csv file and save the clean reviews
data_reordered.to_csv("CleanedData.csv")
# print(new_file)

    
