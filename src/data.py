import pandas as pd
import numpy as np
from keras.preprocessing.text import one_hot
from keras.utils import pad_sequences


def import_data():
    np.random.seed(123)
    data = pd.concat(map(pd.read_csv, ['../dataset/GB_youtube_trending_data.csv',
                                       '../dataset/US_youtube_trending_data.csv']), ignore_index=True)

    # subset the data
    rand_video_ids = np.random.choice(data['video_id'].unique(),
                                     size=int(len(data['video_id'].unique())*0.001),
                                     replace=False)

    data = data.loc[data['video_id'].isin(rand_video_ids)]
    data = data.drop_duplicates(subset='video_id', keep="first")

    data['view_count'] = data['view_count'].where(data['view_count'] > 200000, 1)
    data['view_count'] = data['view_count'].where(data['view_count'] < 200000, 0)
    return data


def split_tags(df):
    # split each of the strings into a list
    df['tags'] = df['tags'].str.split(pat='|')

    # collect all unique tags from those lists
    tags = set(df['tags'].explode().values)

    # create a new Boolean column for each tag
    for tag in tags:
        df[tag] = [tag in df['tags'].loc[i] for i in df.index]

    print(df)


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

