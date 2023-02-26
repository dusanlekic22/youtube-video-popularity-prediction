import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from datetime import datetime
import time


chromedriver_path = "../../../chromedriver_win32/chromedriver.exe"

get_url_channel_about = lambda channel_id: f"https://www.youtube.com/channel/{channel_id}/about"


# load unique channelID from csv file
def load_data(file_path):
    df = pd.read_csv(file_path) #185390
    df_unique_channel_id = df.drop_duplicates(subset=["channelId"], keep='first') #7017
    return list(df_unique_channel_id['channelId'])


def num_there(s):
    return any(i.isdigit() for i in s)


def get_channel_about(channelId, driver, wait):
    channel_subscribe_count, channel_view_count, channel_join_date = 0, 0, '0'

    driver.get(get_url_channel_about(channelId))

    html_subscribe_count = wait.until(EC.presence_of_element_located((By.ID, "subscriber-count")))
    channel_subscribe_count_text = html_subscribe_count.text
    if channel_subscribe_count_text != '' and num_there(channel_subscribe_count_text) and channel_subscribe_count_text != 'Subscriber count hidden':
        channel_subscribe_count_text = channel_subscribe_count_text.replace(' subscribers', '')
        channel_subscribe_count_text = channel_subscribe_count_text.replace(',', '')
        channel_subscribe_count_text = channel_subscribe_count_text.replace('M', '*1000000')
        channel_subscribe_count_text = channel_subscribe_count_text.replace('K', '*1000')
        channel_subscribe_count = int(eval(channel_subscribe_count_text))
    else:
        channel_subscribe_count = -1

    html_view_right_column = wait.until(EC.presence_of_element_located((By.ID, "right-column")))
    channel_about_metadata = html_view_right_column.text.split('\n')
    for metadata in channel_about_metadata:
        if 'Joined' in metadata:
            channel_join_date = metadata.replace('Joined ', '')
        elif 'views' in metadata:
            channel_view_count = metadata.replace(' views', '')
            channel_view_count = int(channel_view_count.replace(',', ''))

    now = datetime.now()
    now_str = now.strftime("%d.%m.%Y %H:%M:%S")
    return [channelId, channel_subscribe_count, channel_view_count, channel_join_date, now_str]


def save_to_csv(channel_about):
    df = pd.DataFrame(channel_about, columns=['channelId', 'channel_subscribe_count', 'channel_view_count', 'channel_join_date', 'scraping_time'])
    df.to_csv('../dataset/US_channel_about.csv', mode='a', header=False, index=False)


def main():
    channelIds = load_data('../dataset/US_youtube_trending_data.csv')
    driver = webdriver.Chrome(chromedriver_path)
    wait = WebDriverWait(driver, 10)

    i = 0
    start_time = time.time()
    channel_about_list = []
    for channelId in channelIds:
        channel_about = get_channel_about(channelId, driver, wait)
        channel_about_list.append(channel_about)
        i += 1
        print(i, channel_about)
        if i % 100 == 0:
            print('100x time:', time.time() - start_time)
            save_to_csv(channel_about_list)
            channel_about_list = []
        if i == 100:
            break

    print('time elapsed: ', time.time() - start_time)
    driver.close()



if __name__ == "__main__":
    main()


