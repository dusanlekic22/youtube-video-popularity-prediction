import json

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sb
from sklearn.preprocessing import OneHotEncoder


def import_data():
    np.random.seed(123)
    us_data = pd.read_csv('../dataset/US_youtube_trending_data.csv', index_col=False)
    gb_data = pd.read_csv('../dataset/GB_youtube_trending_data.csv', index_col=False)

    us_category_dict = category_id_to_category('../dataset/US_category_id.json')
    us_data['categoryId'] = us_data[['categoryId']].apply(convert_category_id_to_category, args=(us_category_dict,),
                                                          axis=1)

    gb_category_dict = category_id_to_category('../dataset/GB_category_id.json')
    gb_data['categoryId'] = gb_data[['categoryId']].apply(convert_category_id_to_category, args=(gb_category_dict,),
                                                          axis=1)

    data = pd.concat([us_data, gb_data])

    us_channel_info = pd.read_csv('../dataset/US_channel_about.csv', index_col=False)
    data = data.merge(us_channel_info, how='left', on='channelId')

    # subset the data
    rand_video_ids = np.random.choice(data['video_id'].unique(),
                                      size=int(len(data['video_id'].unique()) * 1.0),
                                      replace=False)

    data = data.loc[data['video_id'].isin(rand_video_ids)]
    return data


def preprocessing_data(data):
    data = data.drop_duplicates(subset='video_id', keep="first")

    data['trending_time'] = pd.to_datetime(data['trending_date']) - pd.to_datetime(data['publishedAt'])
    data['trending_time'] = data['trending_time'].dt.total_seconds()

    data['channel_subscribe_count'] = data['channel_subscribe_count'].fillna(data['channel_subscribe_count'].median())
    data['channel_view_count'] = data['channel_view_count'].fillna(data['channel_view_count'].median())

    data = category_encoding(data)

    data['view_count'] = data['view_count'].where(data['view_count'] > 400000, 1)
    data['view_count'] = data['view_count'].where(data['view_count'] < 400000, 0)
    return data


def category_encoding(data):
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    oh_cols_train = pd.DataFrame(oh_encoder.fit_transform(data['categoryId'].values.reshape(-1, 1)))
    column_name = oh_encoder.get_feature_names_out(['category'])
    oh_cols_train.columns = column_name
    oh_cols_train.index = data.index

    num_data = data.drop('categoryId', axis=1)
    data = pd.concat([num_data, oh_cols_train], axis=1)

    return data


def category_id_to_category(path):  # creates a dictionary that maps category_id to category name
    category_id_to_category_dict = {}
    with open(path, 'r') as f:
        data = json.load(f)
        for category in data['items']:
            category_id_to_category_dict[category['id']] = category['snippet']['title'].lower().replace(' ', '_')
    return category_id_to_category_dict


def convert_category_id_to_category(id_column, category_dictionary):
    # Using the category to category Id dictionary returns the category name for the given category ID
    category_id = str(id_column[0])
    try:
        category = category_dictionary[category_id]
    except Exception as e:
        category = "none"
    return category


def convert_channel_id_to_channel_title(columns, channel_dictionary):
    # Using the channel Dictionary converts the given channel ID to its Channel Name
    channel_id = columns[0]
    try:
        temp = channel_dictionary[channel_id]
        channel_name = temp['channelTitle']
    except Exception as e:
        channel_name = "none"
    return channel_name


def eda(df):
    print(df.describe())
    print(df.iloc[:, np.r_[7:10, 14:30]].columns)
    sb.set(font_scale=2)
    sb.pairplot(df.iloc[:, np.r_[7:10]], hue="view_count", diag_kind="hist", aspect=2)
    plt.show()
    for column in ['likes', 'dislikes', 'comment_count', 'trending_time', 'view_count']:
        sb.displot(df, x=column, hue="view_count", height=10, aspect=2, multiple="dodge")
        plt.show()


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

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(videos[['likes', 'dislikes', 'comment_count', 'trending_time',
                                                 'channel_subscribe_count', 'channel_view_count']])
    videos[['likes', 'dislikes', 'comment_count', 'trending_time', 'channel_subscribe_count', 'channel_view_count']] \
        = scaled_values

    train_videos = videos.sample(frac=0.8, random_state=200)
    test_videos = videos.drop(train_videos.index)

    return train_videos, test_videos


def split_input_output(train, test):
    x_train = train.drop(['view_count'], axis=1)
    y_train = train[['view_count']]

    x_test = test.drop(['view_count'], axis=1)
    y_test = test[['view_count']]

    return x_train, y_train, x_test, y_test


# preprocessing_data(import_data())
# eda(preprocessing_data(import_data()))
