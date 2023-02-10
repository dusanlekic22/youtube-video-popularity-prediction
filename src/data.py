import pandas as pd
import numpy as np


def import_data():
    np.random.seed(123)
    data = pd.concat(map(pd.read_csv, ['../dataset/GB_youtube_trending_data.csv',
                                       '../dataset/US_youtube_trending_data.csv']), ignore_index=True)

    # subset the data
    rand_video_ids = np.random.choice(data['video_id'].unique(),
                                     size=int(len(data['video_id'].unique()) * 0.01),
                                     replace=False)

    data = data.loc[data['video_id'].isin(rand_video_ids)]
    print(data.shape[0])
    data = data.drop_duplicates(subset='video_id', keep="first")
    print(data.shape[0])

    return data


def split_data(videos):
    # split the train/test split by the latest rating
    train_videos = videos.sample(frac=0.8, random_state=200)
    test_videos = videos.drop(train_videos.index)

    return train_videos, test_videos


def split_input_output(train, test):
    x_train = train.drop(['view_count'], axis=1)
    y_train = train[['view_count']]

    x_test = test.drop(['view_count'], axis=1)
    y_test = test[['view_count']]

    return x_train, y_train, x_test, y_test

