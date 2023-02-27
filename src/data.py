import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import seaborn as sb
from sklearn.preprocessing import OneHotEncoder

def import_data():
    np.random.seed(123)
    data = pd.concat(map(pd.read_csv, ['../dataset/GB_youtube_trending_data.csv',
                                       '../dataset/US_youtube_trending_data.csv']), ignore_index=True)

    # subset the data
    rand_video_ids = np.random.choice(data['video_id'].unique(),
                                     size=int(len(data['video_id'].unique())*1.0),
                                     replace=False)

    data = data.loc[data['video_id'].isin(rand_video_ids)]
    data = data.drop_duplicates(subset='video_id', keep="first")

    data['trending_time'] = pd.to_datetime(data['trending_date']) - pd.to_datetime(data[
        'publishedAt'])
    data['trending_time'] = data['trending_time'].dt.total_seconds()

    OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
    OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(data['categoryId'].values.reshape(-1, 1)))
    column_name = OH_encoder.get_feature_names_out(['category'])
    OH_cols_train.columns = column_name
    OH_cols_train.index = data.index
    print(column_name)
    num_data = data.drop('categoryId', axis=1)
    data = pd.concat([num_data, OH_cols_train], axis=1)
    print(data)
    data['view_count'] = data['view_count'].where(data['view_count'] > 200000, 1)
    data['view_count'] = data['view_count'].where(data['view_count'] < 200000, 0)
    return data


def eda(df):
    print(df.describe())
    print(df.iloc[:, np.r_[7:11, 15:31]].columns)
    sb.set(font_scale=2)
    sb.pairplot(df.iloc[:, np.r_[7:11, 15:31]], hue="view_count", diag_kind="hist", aspect=2)
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
    scaled_values = scaler.fit_transform(videos[['likes', 'dislikes', 'comment_count', 'trending_time']])
    videos[['likes', 'dislikes', 'comment_count', 'trending_time']] = scaled_values

    train_videos = videos.sample(frac=0.8, random_state=200)
    test_videos = videos.drop(train_videos.index)

    return train_videos, test_videos


def split_input_output(train, test):
    x_train = train.drop(['view_count'], axis=1)
    y_train = train[['view_count']]

    x_test = test.drop(['view_count'], axis=1)
    y_test = test[['view_count']]

    return x_train, y_train, x_test, y_test

#eda(import_data())