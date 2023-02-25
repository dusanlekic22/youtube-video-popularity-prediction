import csv
import os

import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
#nltk.download('vader_lexicon')
analyzer = SentimentIntensityAnalyzer()
file_path = '../dataset/US_comment_analysis.csv'
main_dataset_path = '../dataset/US_youtube_trending_data.csv'


def load_data():
    dfr = pd.read_csv(main_dataset_path)
    df_unique_video_id = dfr.drop_duplicates(subset=["video_id"], keep='first')
    return list(df_unique_video_id['video_id'])


def get_comment_list(vid_id):
    dframe = pd.read_csv('../dataset/US_comments.csv')
    video_comments = []
    for index, row in dframe.iterrows():
        video = row[1]
        if video == vid_id:
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


def save_to_csv(data):

    row = pd.DataFrame({'Video': [data[0]], 'Neutral': [data[1]], 'Negative': [data[2]], 'Positive': [data[3]]})

    df = pd.read_csv(file_path)
    df = pd.concat([df, row], ignore_index=True)

    df.to_csv(file_path, index=False)


def create_csv_if_needed():

    with open(file_path, 'w', newline='') as csvfile:

        writer = csv.writer(csvfile)

        headers = ['Video', 'Neutral', 'Negative', 'Positive']
        writer.writerow(headers)

    csvfile.close()


if __name__ == '__main__':

    video_ids = load_data()
    create_csv_if_needed()

    for i, video_id in enumerate(video_ids):
        print(video_id)
        comments = get_comment_list(video_id)
        neutral, negative, positive = analyse_sentiment(comments)
        print(f'Neutralnih {neutral}, negativnih {negative}, pozitivnih {positive}')
        new_row = video_id, neutral, negative, positive, i+1
        save_to_csv(new_row)
