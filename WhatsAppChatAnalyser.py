import re
import pandas as pd
import matplotlib.pyplot as plt

def startsWithDateTime(s):
    pattern = '([0-2][0-9]|(3)[0-1])(\/)(((0)[0-9])|((1)[0-2]))(\/)(\d\d|\d\d\d\d), ((([0-9][0-9]):([0-9][0-9]))|(([0-9]):([0-9][0-9]))) (am|pm) -'
    result = re.match(pattern, s)
    if result:
        return True
    return False

def startsWithAuthor(s):
    patterns = [
        '([\w]+):',                        # First Name
        '([\w]+[\s]+[\w]+):',              # First Name + Last Name
        '([\w]+[\s]+[\w]+[\s]+[\w]+):',    # First Name + Middle Name + Last Name
        '([+]\d{2} \d{5} \d{5}):',         # Mobile Number (India)
    ]
    pattern = '^' + '|'.join(patterns)
    result = re.match(pattern, s)
    if result:
        return True
    return False

def getDataPoint(line):
    splitLine = line.split(' - ')
    dateTime = splitLine[0] 
    date, time = dateTime.split(', ')
    message = ' '.join(splitLine[1:])
    if startsWithAuthor(message):
        splitMessage = message.split(': ')
        author = splitMessage[0]
        message = ' '.join(splitMessage[1:])
    else:
        author = None
    return date, time, author, message

#%%
# Segregates messages into a dataframe
parsedData = []
conversationPath = 'hdk.txt'
with open(conversationPath, encoding="utf-8") as fp:
    fp.readline()
    messageBuffer = []
    date, time, author = None, None, None
    while True:
        line = fp.readline() 
        if not line:
            parsedData.append([date, time, author, ' '.join(messageBuffer)])
            break
        line = line.strip()
        if startsWithDateTime(line):
            if len(messageBuffer) > 0:
                parsedData.append([date, time, author, ' '.join(messageBuffer)])
            messageBuffer.clear() 
            date, time, author, message = getDataPoint(line) 
            messageBuffer.append(message)
        else:
            messageBuffer.append(line)

df = pd.DataFrame(parsedData, columns=['Date', 'Time', 'Author', 'Message'])
df.head()

#%%
# Frequency of all authors' messages sent
author_value_counts = df['Author'].value_counts()
author_value_counts.plot.barh()
plt.xlabel('Number of Messages')
plt.ylabel('Authors')

# Frequency of all authors who sent media on the group
media_messages_df = df[df['Message'] == '<Media omitted>']
author_media_messages_value_counts = media_messages_df['Author'].value_counts()
author_media_messages_value_counts.plot.barh()
plt.xlabel('Number of Media Messages')
plt.ylabel('Authors')

#%%
# Get rid of non-text data
null_authors_df = df[df['Author'].isnull()]
messages_df = df.drop(null_authors_df.index) # Drop messages with {None} author
messages_df = messages_df.drop(media_messages_df.index) # Drop messages with media sent
messages_df.head()

#%%
# Add letter and word count for each message
messages_df['Word_Count'] = messages_df['Message'].apply(lambda s : len(s.split(' ')))
messages_df['Letter_Count'] = messages_df['Message'].apply(lambda s : len(s))

#%%
# Words sent by each author
total_word_count_grouped_by_author = messages_df[['Author', 'Word_Count']].groupby('Author').sum()
sorted_total_word_count_grouped_by_author = total_word_count_grouped_by_author.sort_values('Word_Count', ascending=False)
sorted_total_word_count_grouped_by_author.plot.barh()
plt.xlabel('Number of Words')
plt.ylabel('Authors')

#%%
# Time of the day when group is most active
messages_df['Time'].value_counts().head(5).plot.barh()
plt.xlabel('Number of Messages')
plt.ylabel('Time')

#%%
# Sentiment analysis of messages
from nltk.sentiment.vader import SentimentIntensityAnalyzer 
sid = SentimentIntensityAnalyzer()

overallSentiments = []
compoundScores = []
for msg in messages_df['Message']:
    ss = sid.polarity_scores(msg)
    compoundScores.append(round(ss['compound'], 5))
    # we'll compare the compound score only - https://github.com/cjhutto/vaderSentiment#about-the-scoring
    if ss['compound'] >= 0.05:
        feel = 'Positive'
    elif ss['compound'] > -0.05 and ss['compound'] < 0.05:
        feel = 'Neutral'
    elif ss['compound'] < -0.05:
        feel = 'Negative'
    overallSentiments.append(feel)
    
messages_df = messages_df.assign(Compound_Scores = compoundScores,
                                 Overall_Sentiment = overallSentiments)

total_sentiment_count_grouped_by_author = messages_df[['Author', 'Compound_Scores']].groupby('Author').sum()
total_messages_count_grouped_by_author = messages_df['Author'].value_counts()
average_user_sentiment = total_sentiment_count_grouped_by_author['Compound_Scores']/total_messages_count_grouped_by_author
total_sentiment_count_grouped_by_author = total_sentiment_count_grouped_by_author.assign(Average_User_Sentiment = average_user_sentiment)
average_user_sentiment.plot.barh()
plt.xlabel('Average Sentiment of Sent Messages')
plt.ylabel('Authors')