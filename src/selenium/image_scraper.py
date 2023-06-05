import requests # request img from web
import shutil # save img locally
import pandas as pd


def download_image(row):
    res = requests.get(row['thumbnail_link'], stream=True)
    i = 0
    file_name = row['video_id'] + '.jpg'
    i += 1
    if res.status_code == 200:
        with open('../../dataset/video_thumbnails/' + file_name, 'wb') as f:
            shutil.copyfileobj(res.raw, f)
        print('Image sucessfully Downloaded: ', file_name, ' ', i)
    else:
        print('Image Couldn\'t be retrieved')


if __name__ == '__main__':
    data = pd.read_csv('../../dataset/US_youtube_trending_data.csv', index_col=False)
    data = data.drop_duplicates(subset='video_id', keep="first")
    for index, row in data.iterrows():
        download_image(row)
