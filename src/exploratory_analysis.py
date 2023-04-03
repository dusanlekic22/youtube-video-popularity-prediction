import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


main_dataset_path = '../dataset/US_youtube_trending_data.csv'
data = pd.read_csv(main_dataset_path)
print(data.dtypes)

#Korelacija pojedinacnih kolona (numerickih) sa kolonom view_count

correlation = data['comment_count'].corr(data['view_count'])
print(f"The correlation between comment_count and view_count is: {correlation}")

correlation = data['likes'].corr(data['view_count'])
print(f"The correlation between likes and view_count is: {correlation}")

correlation = data['dislikes'].corr(data['view_count'])
print(f"The correlation between dislikes and view_count is: {correlation}")

#Zastupljenost kategorija u trendingu

newData = data.groupby("categoryId").size().reset_index(name='count')
plt.figure()
plt.barh(newData["categoryId"],newData["count"])

plt.xlabel("Number of trending videos")
plt.ylabel("Categories")
plt.title("Categories in trending")
plt.show()

#sorted values
dataS = newData.sort_values(by="count",ascending=False)
print(dataS)







