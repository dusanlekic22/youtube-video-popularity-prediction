import json


def get_categories():

    map={}
    #read file
    categories_file = open('../dataset/US_category_id.json','r')
    jsondata = categories_file.read()

    #parse
    obj = json.loads(jsondata)
    items = obj["items"]

    for i in range(len(items)):
        #print(items[i]["id"])
        #print(items[i]["snippet"]["title"])
        map[items[i]["id"]] = items[i]["snippet"]["title"]
    return map