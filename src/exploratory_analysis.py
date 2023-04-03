import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from src.categories import get_categories

main_dataset_path = '../dataset/US_youtube_trending_data.csv'
data = pd.read_csv(main_dataset_path,parse_dates=["publishedAt"])

categories = get_categories()
print(data.dtypes)

def assign_category(id):
    return categories.get(str(id))

data["category"] = data["categoryId"].apply(assign_category)
print(data.head(5))



#Korelacija pojedinacnih kolona (numerickih) sa kolonom view_count

correlation = data['comment_count'].corr(data['view_count'])
print(f"The correlation between comment_count and view_count is: {correlation}")

correlation = data['likes'].corr(data['view_count'])
print(f"The correlation between likes and view_count is: {correlation}")

correlation = data['dislikes'].corr(data['view_count'])
print(f"The correlation between dislikes and view_count is: {correlation}")

#Zastupljenost kategorija u trendingu

newData = data.groupby("category").size().reset_index(name='count')
plt.figure()
plt.barh(newData["category"],newData["count"])

plt.xlabel("Number of trending videos")
plt.ylabel("Categories")
plt.title("Categories in trending")
#plt.show()

#sorted values
dataS = newData.sort_values(by="count",ascending=False)
print(dataS.head(5))

#change over the years

#2020
dataset_2020 = data.loc[data["publishedAt"].dt.year == 2020]
dataset_2020 = dataset_2020.groupby("category").size().reset_index(name='count').sort_values(by="count",ascending=False).head(5)
plt.figure()
plt.barh(dataset_2020["category"], dataset_2020["count"])

plt.xlabel("Trending 2020")
plt.ylabel("Categories")
plt.show()

#2021
dataset_2021 = data.loc[data["publishedAt"].dt.year == 2021]
dataset_2021 = dataset_2021.groupby("category").size().reset_index(name='count').sort_values(by="count",ascending=False).head(5)
plt.figure()
plt.barh(dataset_2021["category"], dataset_2021["count"])

plt.xlabel("Trending 2021")
plt.ylabel("Categories")
plt.show()

#2022
dataset_2022 = data.loc[data["publishedAt"].dt.year == 2022]
dataset_2022 = dataset_2022.groupby("category").size().reset_index(name='count').sort_values(by="count",ascending=False).head(5)
plt.figure()
plt.barh(dataset_2022["category"], dataset_2022["count"])

plt.xlabel("Trending 2022")
plt.ylabel("Categories")
plt.show()


#2023
dataset_2023 = data.loc[data["publishedAt"].dt.year == 2023]
dataset_2023 = dataset_2023.groupby("category").size().reset_index(name='count').sort_values(by="count",ascending=False).head(5)
plt.figure()
plt.barh(dataset_2023["category"], dataset_2023["count"])

plt.xlabel("Trending 2023")
plt.ylabel("Categories")
plt.show()








