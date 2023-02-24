import pandas as pd
from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from datetime import datetime
import time

chromedriver_path = "../../../chromedriver_win32/chromedriver.exe"
driver = webdriver.Chrome(chromedriver_path)
wait = WebDriverWait(driver, 10)


# load unique video ids from csv file
def load_data(file_path):
    df = pd.read_csv(file_path)
    df_unique_video_id = df.drop_duplicates(subset=["video_id"], keep='first')
    return list(df_unique_video_id['video_id'])


def is_comment_empty_or_blank(comment):
    if comment is None:
        return True
    if comment == '':
        return True
    if comment.isspace():
        return True
    return False


def preprocess_comment(comment):
    comment = comment.strip()
    comment = comment.replace('\n', ' ')
    comment = comment.replace('\r', ' ')
    comment = comment.replace('\t', ' ')
    return comment



def get_comments(video_id):
    driver.get('https://www.youtube.com/watch?v=' + video_id)
    driver.maximize_window()

    html = driver.find_element(By.TAG_NAME, 'html')
    for i in range(10):
        html.send_keys(Keys.END)
        time.sleep(0.5)

    # wait for comments to load
    html_comment_section = wait.until(EC.presence_of_element_located((By.CLASS_NAME, "style-scope ytd-item-section-renderer")))
    comment_renderer_tag = html_comment_section.find_elements_by_tag_name("ytd-comment-thread-renderer")

    target_num_of_comments = 50
    num_of_comments = 0
    comments_list = []
    now = datetime.now()
    now_str = now.strftime("%d.%m.%Y %H:%M:%S")
    for comment_renderer in comment_renderer_tag:
        comment = comment_renderer.find_element_by_xpath('.//*[@id="content-text"]')
        creator_heart = None
        try:
            creator_heart = comment_renderer.find_element_by_xpath('.//*[@id="creator-heart-button"]')
        except:
            pass
        if is_comment_empty_or_blank(str(comment.text)):
            continue
        comments_list.append((video_id, int(creator_heart is not None) , preprocess_comment(str(comment.text)), now_str))
        num_of_comments += 1
        if num_of_comments == target_num_of_comments:
            break

    #driver.close()
    for c in comments_list:
        print(c)
    print(len(comments_list))
    return comments_list


def save_to_csv(comments_list):
    if len(comments_list) == 0:
        return
    df = pd.DataFrame(comments_list, columns=['video_id', 'creator_heart', 'comment', 'date'])
    df.to_csv('../dataset/US_comments.csv', mode='a', header=False)


def main():
    video_ids = load_data('../dataset/US_youtube_trending_data.csv')
    print(len(video_ids))
    exit()
    n = 0
    elapsed_time = 0
    for video_id in video_ids:
        n += 1
        print(video_id)
        start_time = time.time()
        comment_list = get_comments(video_id)
        save_to_csv(comment_list)
        end_time = time.time()
        elapsed_time += end_time - start_time

        print("---", n, "-------------------------------------------------", end_time - start_time, "seconds ---", elapsed_time, "seconds ---")
        if n == 10:
            break



if __name__ == '__main__':
    main()
    driver.close()

