import csv
#from textblob import TextBlob
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()


comments_filepath = 'dataset/US_comments.csv'
comment_polarity_error = 0.25
sentiment_analysers = ['vader', 'textblob']
sentiment_analyser = sentiment_analysers[0]

def load_comments():
    df = pd.read_csv(comments_filepath)
    return df


def main():
    df = load_comments()
    #print(df.iloc[0])
    current_video_id = '3C66w5Z0ixs'

    def save_to_csv(comment_analysis_result):
        df = pd.DataFrame(comment_analysis_result, columns=['video_id', 'number_of_comments', 'number_of_positive_comments', 'number_of_negative_comments', 'number_of_neutral_comments'])
        df.to_csv('dataset/US_comments_analysis_vader.csv', mode='a', header=False, index=False)

    number_of_comments = 0
    number_of_positive_comments = 0
    number_of_negative_comments = 0
    number_of_neutral_comments = 0

    comment_polarity = []
    for index, row in df.iterrows():
        n, video_id, creator_liked, comment, date_of_download = row
        print(index, n, video_id)

        if video_id != current_video_id:
            save_to_csv([(current_video_id, number_of_comments, number_of_positive_comments, number_of_negative_comments, number_of_neutral_comments)])
            current_video_id = video_id
            comment_polarity = []
            number_of_comments = 0
            number_of_positive_comments = 0
            number_of_negative_comments = 0
            number_of_neutral_comments = 0

        number_of_comments += 1

        polarity = 0
        if sentiment_analyser == 'vader':
            scores = analyzer.polarity_scores(comment)
            polarity = scores['compound']
        elif sentiment_analyser == 'textblob':
            #polarity = TextBlob(str(comment)).sentiment.polarity
            pass


        if polarity > comment_polarity_error:
            number_of_positive_comments += 1
        elif polarity < -comment_polarity_error:
            number_of_negative_comments += 1
        else:
            number_of_neutral_comments += 1

        comment_polarity.append((comment, polarity))


if __name__ == '__main__':
    main()