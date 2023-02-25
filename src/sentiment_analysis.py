import pandas as pd


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


#def analyse_sentiment(comments):


if __name__ == '__main__':
    print('Printaj sve')
    video_ids = load_data('../dataset/US_youtube_trending_data.csv')

    for video_id in video_ids:
        print(video_id)
        comments = get_comment_list(video_id)

        for c in comments:
            print(c)
        break
        #nautral, positive, negative = analyse_sentiment(comments)
        #serijalizuj u file
