import tweepy
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud

# Set up Twitter API credentials (replace with your own keys)
consumer_key = 'YOUR_CONSUMER_KEY'
consumer_secret = 'YOUR_CONSUMER_SECRET'
access_token = 'YOUR_ACCESS_TOKEN'
access_token_secret = 'YOUR_ACCESS_TOKEN_SECRET'

# Authenticate with Twitter API
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth)

# Define the search term (e.g., a specific hashtag, topic, or brand)
search_term = 'Apple'

# Collect tweets (e.g., 1000 tweets) using the updated method `search_tweets`
tweets = tweepy.Cursor(api.search_tweets, q=search_term, lang='en', tweet_mode='extended').items(1000)

# Store tweets in a DataFrame
data = pd.DataFrame([tweet.full_text for tweet in tweets], columns=['Tweet'])

# Perform Sentiment Analysis using TextBlob
def get_sentiment(text):
    analysis = TextBlob(text)
    # Sentiment polarity: -1 is negative, 1 is positive, and 0 is neutral
    return analysis.sentiment.polarity

# Apply the sentiment function to each tweet
data['Sentiment'] = data['Tweet'].apply(get_sentiment)

# Classify sentiment as Positive, Negative, or Neutral based on polarity
def sentiment_label(polarity):
    if polarity > 0:
        return 'Positive'
    elif polarity < 0:
        return 'Negative'
    else:
        return 'Neutral'

data['Sentiment_Label'] = data['Sentiment'].apply(sentiment_label)

# 1. Sentiment Distribution Bar Plot
sentiment_counts = data['Sentiment_Label'].value_counts()

plt.figure(figsize=(8, 6))
sentiment_counts.plot(kind='bar', color=['green', 'red', 'blue'])
plt.title(f'Sentiment Distribution for Tweets about {search_term}')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.show()

# 2. Word Cloud for Positive and Negative Sentiments
positive_tweets = ' '.join(data[data['Sentiment_Label'] == 'Positive']['Tweet'])
negative_tweets = ' '.join(data[data['Sentiment_Label'] == 'Negative']['Tweet'])

positive_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(positive_tweets)
negative_wordcloud = WordCloud(width=800, height=400, background_color='white').generate(negative_tweets)

plt.figure(figsize=(12, 6))

# Display Positive Word Cloud
plt.subplot(1, 2, 1)
plt.imshow(positive_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Positive Sentiment')
plt.axis('off')

# Display Negative Word Cloud
plt.subplot(1, 2, 2)
plt.imshow(negative_wordcloud, interpolation='bilinear')
plt.title('Word Cloud for Negative Sentiment')
plt.axis('off')

plt.show()

# 3. Sentiment Trend Over Time
# Add datetime information to the DataFrame
data['Datetime'] = pd.to_datetime([tweet.created_at for tweet in tweets])

# Group by day and calculate average sentiment per day
data['Date'] = data['Datetime'].dt.date
daily_sentiment = data.groupby('Date')['Sentiment'].mean()

plt.figure(figsize=(10, 6))
daily_sentiment.plot(kind='line', color='blue')
plt.title(f'Trend of Sentiment for {search_term} Over Time')
plt.xlabel('Date')
plt.ylabel('Average Sentiment')
plt.xticks(rotation=45)
plt.show()
