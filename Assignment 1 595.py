from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import pandas as pd
import snscrape.modules.twitter as tw
import requests
from datetime import datetime as dt
import pytz
import nltk
from nltk import corpus
from nltk.corpus import stopwords
nltk.download('vader_lexicon')
nltk.download('wordnet')
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import typing
import pandas_datareader
from pandas_datareader import data
from bs4 import BeautifulSoup


#Part 2

michael_company_data = pd.read_csv('C:/Users/ekim1_hxy84z7/Documents/Year 5/595/fake_company_pull.csv')
brett_company_pull = pd.read_csv('C:/Users/ekim1_hxy84z7/Documents/Year 5/595/CompanyPull.csv')
stephen_company_pull = pd.read_csv('C:/Users/ekim1_hxy84z7/Documents/Year 5/595/comps.csv')


df = pd.DataFrame(columns=['Title', 'Purpose'])
df = df.append(michael_company_data)
df = df.append(brett_company_pull)
df = df.append(stephen_company_pull)
df.rename(columns={'Unnamed: 0':'axis'}, inplace=True )
df.set_index('axis', inplace=True)
df[['junk','Purpose']] = df.Purpose.str.split(": ", expand=True)
df.drop(['junk'], axis=1, inplace=True)

#Part 3

stopcorpus: typing.List = stopwords.words('english')
def remove_words(em:str,list_of_words_to_remove: typing.List):
    return [item for item in em if item not in list_of_words_to_remove]
def collapse_list_to_string(string_list):
    return ' '.join(string_list)
def remove_apostrophes(text):
    text = text.replace("'", "")
    text = text.replace('"', "")
    text = text.replace('`', "")
    return text

df['Purpose'] = df['Purpose'].astype(str).apply(lambda x: remove_words(x.split(),stopcorpus))
df['Purpose'] = df['Purpose'].apply(collapse_list_to_string)
df['Purpose'] = df['Purpose'].apply(remove_apostrophes)

tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()
def root_word(text):
    return [lemmatizer.lemmatize(w) for w in tokenizer.tokenize(text)]

df['Purpose'] = df['Purpose'].astype(str).apply(root_word)
df['Purpose'] = df['Purpose'].apply(collapse_list_to_string)


sentiment = []
analyzer = SentimentIntensityAnalyzer()
for purpose in df.Purpose:
    vs = analyzer.polarity_scores(purpose)
    sentiment.append(vs["compound"])
df['Sentiment'] = sentiment


df.sort_values(by=['Sentiment'], inplace=True)
worst_sentiment = df.head(5)
best_sentiment = df.tail(5)
np.savetxt(r'C:/Users/ekim1_hxy84z7/Documents/Year 5/595/best_values.txt', best_sentiment.values, fmt='%s')
np.savetxt(r'C:/Users/ekim1_hxy84z7/Documents/Year 5/595/worst_values.txt', worst_sentiment.values, fmt='%s')
print(best_sentiment)
print(worst_sentiment)



print("done")