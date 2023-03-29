import json

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
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
    data = us_data
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

    print(data['dislikes'].value_counts())

    data = replace_outliers(data, ['likes', 'dislikes', 'comment_count',
                                   'channel_subscribe_count', 'channel_view_count', 'trending_time'])
    data['dislikes'] = data['dislikes'].where(data['dislikes'] != 0, data['dislikes'].median())

    data['channel_subscribe_count'] = data['channel_subscribe_count'].fillna(data['channel_subscribe_count'].median())
    data['channel_view_count'] = data['channel_view_count'].fillna(data['channel_view_count'].median())

    data = category_encoding(data)
    data = split_tags(data)

    # replace view count column with 0 where view count is less than 400000 and 1 where view count is greater than 400000

    data['view_count'] = data['view_count'].where(data['view_count'] < 500000, 1)
    data['view_count'] = data['view_count'].where(data['view_count'] == 1, 0)

    return data


def replace_outliers(data, columns_name):
    for column_name in columns_name:
        median = data[column_name].median()
        q1 = data[column_name].quantile(0.25)
        q3 = data[column_name].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)
        data[column_name] = data[column_name].where(data[column_name] > lower_bound, lower_bound)
        data[column_name] = data[column_name].where(data[column_name] < upper_bound, upper_bound)
        # data[column_name] = data[column_name].where(data[column_name] > lower_bound, median)
        # data[column_name] = data[column_name].where(data[column_name] < upper_bound, median)
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


# method for tag encoding
def tags_encoding(data):
    oh_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    oh_cols_train = pd.DataFrame(oh_encoder.fit_transform(data['tags'].values.reshape(-1, 1)))
    column_name = oh_encoder.get_feature_names_out(['tags'])
    oh_cols_train.columns = column_name
    oh_cols_train.index = data.index

    num_data = data.drop('tags', axis=1)
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
    df['dislikes'].hist()
    plt.show()
    print(df.iloc[:, 0:17].describe())
    print(df.iloc[:, np.r_[7:10, 14:30]].columns)
    sb.set(font_scale=1)
    sb.pairplot(df.iloc[:, np.r_[6:9, 14:16]], hue="view_count", diag_kind="hist", aspect=2)
    plt.show()
    for column in ['likes', 'dislikes', 'comment_count', 'trending_time', 'view_count', 'channel_subscribe_count',
                   'channel_view_count']:
        sb.displot(df, x=column, hue="view_count", height=10, aspect=2, multiple="dodge")
        plt.show()


def split_tags(df):
    # split each of the strings into a list
    df['tags'] = df['tags'].str.split(pat='|')
    # make a dictionary of all the tags and their counts
    tag_counts = {}
    for tags in df['tags']:
        for tag in tags:
            if tag in tag_counts:
                tag_counts[tag] += 1
            else:
                tag_counts[tag] = 1
    # sort the dictionary by the counts
    tag_counts = {k: v for k, v in sorted(tag_counts.items(), key=lambda item: item[1], reverse=True)}

    # create a list of the top 100 tags
    top_tags = list(tag_counts.keys())[:20]

    # create a new column that is a list of the top 100 tags
    df['tags'] = df['tags'].apply(lambda x: [item for item in x if item in top_tags])

    mlb = MultiLabelBinarizer(sparse_output=True)

    df = df.join(
        pd.DataFrame.sparse.from_spmatrix(
            mlb.fit_transform(df.pop('tags')),
            index=df.index,
            columns='tag_'+mlb.classes_))

    df.rename(columns={'tag_[None]': 'tag_none'}, inplace=True)

    return df

def split_data(videos):
    # split the train/test split by the latest rating

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(videos[['likes', 'comment_count', 'trending_time',
                                                 'channel_subscribe_count', 'channel_view_count']])
    videos[['likes', 'comment_count', 'trending_time', 'channel_subscribe_count', 'channel_view_count']] \
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


#preprocessing_data(import_data())
#eda(preprocessing_data(import_data()))
