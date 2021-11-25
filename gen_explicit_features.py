import csv
import pandas as pd
import re


def data_length(data):
    data = data.to_frame(name="text")
    lengths = []
    url_count = []
    for _, row in data.iterrows():
        text = row["text"]
        lengths.append(len(text))
        regex = r"(?i)\b((?:https?://|www\d{0,3}[.]|[a-z0-9.\-]+[.][a-z]{2,4}/)(?:[^\s()<>]+|\(([^\s()<>]+|(\([^\s()<>]+\)))*\))+(?:\(([^\s()<>]+|(\([^\s()<>]+\)))*\)|[^\s`!()\[\]{};:'\".,<>?«»“”‘’]))"
        url = re.findall(regex, text)
        url_count.append(len(url))
    return lengths, url_count


f = "./followers.csv"
d = pd.read_csv(f, sep=",")
followers = d["Followers"]
f = "dataset_final.csv"
data = pd.read_csv(f, sep=",")
data_ids = data["Id"].to_list()
lengths, url_count = data_length(data["Text"].astype("str"))
writing_data1 = pd.DataFrame(columns=["text_length", "url_count", "followers"])
print("reached")
for ind in range(len(followers)):
    writing_data1.append(
        {
            "text_length": lengths[ind],
            "url_count": url_count[ind],
            "followers": followers[ind],
        },
        ignore_index=True,
    )
writing_data1.to_csv("explicit_features.csv", index=False, header=True)
