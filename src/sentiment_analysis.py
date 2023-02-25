import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()


def load_data(file_path):
    df = pd.read_csv(file_path)
    df_unique_video_id = df.drop_duplicates(subset=["video_id"], keep='first')
    return list(df_unique_video_id['video_id'])


def get_comment_list(video_id):
    df = pd.read_csv('../dataset/US_comments.csv')
    video_comments = []
    for index, row in df.iterrows():
        video = row[1]
        if video == video_id:
            video_comments.append(row[3])

    return video_comments


def analyse_sentiment(comment_list):

    neutral_count = 0
    negative_count = 0
    positive_count = 0

    for comment in comment_list:
        scores = analyzer.polarity_scores(comment)
        compound_score = scores['compound']
        if compound_score >= 0.05:
            positive_count = positive_count + 1
        elif compound_score <= -0.05:
            negative_count = negative_count + 1
        else:
            neutral_count = neutral_count + 1

    return neutral_count, negative_count, positive_count


if __name__ == '__main__':

    video_ids = load_data('../dataset/US_youtube_trending_data.csv')

    for video_id in video_ids:
        print(video_id)
        comments = get_comment_list(video_id)
        neutral, negative, positive = analyse_sentiment(comments)
        print(f'Neutralnih {neutral}, negativnih {negative}, pozitivnih {positive}')
        break
