import tweepy
from textblob import TextBlob

consumer_key = '0bshlWVSX0O0DnU32gjQWSz5X'
consumer_secret = 'aVlvqZwSKPShPMq4k7cwTuXXEyjlUzNCfOesy1dOyrL7Gu6NW9'
access_token = '1259307926-z5WCKEGu0UXBkBTFHLACQMX20BJeSq5YQIReWO1'
acces_token_secret = 'zrRDk2LhcHTvs04nT7UNYaPFsUk3M27hB5a5bcOdHymT3'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, acces_token_secret)

api = tweepy.API(auth)
public_tweets = api.search('Trump')

for tweet in public_tweets:
    print(tweet.text)
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
