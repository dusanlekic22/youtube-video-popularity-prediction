import json

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from nltk import WordNetLemmatizer
from scipy import stats
from sklearn.preprocessing import MinMaxScaler, MultiLabelBinarizer
import seaborn as sb
from sklearn.preprocessing import OneHotEncoder
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
def import_data():
    np.random.seed(123)
    us_data = pd.read_csv('../dataset/US_youtube_trending_data.csv', index_col=False)

    us_category_dict = category_id_to_category('../dataset/US_category_id.json')
    us_data['categoryId'] = us_data[['categoryId']].apply(convert_category_id_to_category, args=(us_category_dict,),
                                                          axis=1)

    data = us_data
    us_channel_info = pd.read_csv('../dataset/US_channel_about.csv', index_col=False)
    data = data.merge(us_channel_info, how='left', on='channelId')
    data = data.drop(['channelId', 'channelTitle', 'thumbnail_link', 'comments_disabled', 'ratings_disabled',
                      'channel_join_date', 'scraping_time'], axis=1)
    us_vader_sentiment = pd.read_csv('../dataset/US_comments_analysis_vader_full.csv', index_col=False)
    #set number of positive, negative and neutral comments to a percentage of total number of comments
    #us_textblob_sentiment['number_of_positive_comments'] = us_textblob_sentiment['number_of_positive_comments'] / us_textblob_sentiment['number_of_comments']
    #us_textblob_sentiment['number_of_negative_comments'] = us_textblob_sentiment['number_of_negative_comments'] / us_textblob_sentiment['number_of_comments']
    #us_textblob_sentiment['number_of_neutral_comments'] = us_textblob_sentiment['number_of_neutral_comments'] / us_textblob_sentiment['number_of_comments']

    #drop id and number of comments columns
    us_textblob_sentiment = us_vader_sentiment.drop(['number_of_comments'], axis=1)

    # merge the sentiment data
    data = data.merge(us_textblob_sentiment, how='left', on='video_id')
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
    #drop trending date and published at columns
    data = data.drop(['trending_date', 'publishedAt', 'video_id'], axis=1)
    #create like, dislike an
    # replace the likes column with the like to view ratio
    # data['like_ratio'] = data['likes'] / data['view_count']
    # # replace the dislikes column with the dislike to view ratio
    # #data['dislike_ratio'] = data['dislikes'] / data['view_count']
    # # replace the comment count column with the comment count to view ratio
    # data['comment_count_ratio'] = data['comment_count'] / data['view_count']
    # replace infinity values with 0
    #data = data.replace([np.inf, -np.inf], 0)

    data = replace_outliers(data, ['likes', 'dislikes', 'comment_count',
                                   'channel_subscribe_count', 'channel_view_count', 'trending_time'])
    data['dislikes'] = data['dislikes'].where(data['dislikes'] != 0, data['dislikes'].median())

    data['channel_subscribe_count'] = data['channel_subscribe_count'].fillna(data['channel_subscribe_count'].median())
    data['channel_view_count'] = data['channel_view_count'].fillna(data['channel_view_count'].median())

    #filling missing values with median for number of negative, neutral and positive comments
    data['number_of_negative_comments'] = data['number_of_negative_comments'].fillna(data['number_of_negative_comments'].median())
    data['number_of_neutral_comments'] = data['number_of_neutral_comments'].fillna(data['number_of_neutral_comments'].median())
    data['number_of_positive_comments'] = data['number_of_positive_comments'].fillna(data['number_of_positive_comments'].median())

    data = category_encoding(data)
    #data = split_tags(data)

    #print column names and their index
    # for i, col in enumerate(data.columns):
    #     print(i, col)


    # split the view count column into 3 balanced integer categories
    data['view_count'] = pd.qcut(data['view_count'], 2, labels=[0, 1])


    #print how many values are in view count column
    print(data['view_count'].value_counts())

    scaler = MinMaxScaler()
    scaled_values = scaler.fit_transform(data[['likes', 'dislikes', 'comment_count', 'trending_time',
                                                 'channel_subscribe_count', 'channel_view_count',
                                                 'number_of_positive_comments', 'number_of_negative_comments',
                                                 'number_of_neutral_comments', 'view_count']])
    data[['likes', 'dislikes', 'comment_count', 'trending_time', 'channel_subscribe_count', 'channel_view_count',
            'number_of_positive_comments', 'number_of_negative_comments', 'number_of_neutral_comments', 'view_count']] \
        = scaled_values

    #drop number of positive, negative and neutral comments columns

    data = remove_stop_words(data)
    data = lemmatize_words(data)
    #remove '|' from title and description
    data['title'] = data['title'].str.replace('|', '')
    data['description'] = data['description'].str.replace('|', '')
    data['tags'] = data['tags'].str.replace('|', '')
    #remove '-' from title and description
    data['title'] = data['title'].str.replace('-', '')
    data['description'] = data['description'].str.replace('-', '')
    data['tags'] = data['tags'].str.replace('-', '')
    #set to lower case
    data['title'] = data['title'].str.lower()
    data['description'] = data['description'].str.lower()
    data['tags'] = data['tags'].str.lower()

    return data

def remove_stop_words(data):
    stop_words = set(stopwords.words('english'))
    #print data description type
    # print(type(data['description']))
    data['title'] = data['title'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words)]))
    data['description'] = data['description'].apply(lambda x: ' '.join([word for word in str(x).split() if word not in (stop_words)]))
    return data

def lemmatize_words(data):
    lemmatizer = WordNetLemmatizer()
    data['title'] = data['title'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
    data['description'] = data['description'].apply(lambda x: ' '.join([lemmatizer.lemmatize(word) for word in x.split()]))
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
    #drop positive,negative, neutral number of comments
    #videos = videos.drop(['number_of_positive_comments', 'number_of_negative_comments', 'number_of_neutral_comments'], axis=1)\

    train_videos = videos.sample(frac=0.8, random_state=200)
    test_videos = videos.drop(train_videos.index)

    return train_videos, test_videos


def split_input_output(train, test):
    x_train = train.drop(['view_count'], axis=1)
    y_train = train[['view_count']]

    x_test = test.drop(['view_count'], axis=1)
    y_test = test[['view_count']]

    return x_train, y_train, x_test, y_test

#eda(preprocessing_data(import_data()))
