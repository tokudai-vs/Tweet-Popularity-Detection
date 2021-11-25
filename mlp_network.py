# from sklearn.neural_network import MLPRegressor
from nltk.stem.lancaster import LancasterStemmer
from nltk.corpus import stopwords
import re
from operator import itemgetter


def prepareSentence(s):
    stemmer = LancasterStemmer()
    ignore_words = set(stopwords.words("english"))
    regpattern = re.compile('[\W_]+" "')
    s = re.sub("[^A-Za-z ]+", "", s)
    words = nltk.word_tokenize(s.lower())
    return [stemmer.stem(w.lower()) for w in words if w not in ignore_words]


import pandas as pd
import numpy as np
import random

np.random.seed(0)
random.seed(9)
# tf.random.set_seed(123)
from numpy import array
import re
import nltk

# nltk.download('stopwords')
from ast import literal_eval

pd.set_option("display.max_colwidth", None)


def vocab_count(dataframe):
    # Get pandas dataframe and calculate vocablury size
    results = set()
    dataframe["Text"].str.lower().str.split().apply(results.update)
    return len(results)


def retweet_class_count(data):
    results = set()
    for l in data:
        results.update(l)
    # results = set(filter(lambda x:isInt(x), results))
    return max(results) + 1


def data_cleaner(data):
    from nltk.corpus import stopwords

    # Clean data and remove
    # TODO
    # This is neccessary as we're getting random numbers in final data. Couldn't find the cause.
    data = data.to_frame(name="text")
    data_cleaned = []  # List as we need it to be fed to model

    stop_words = set(stopwords.words("english"))
    whitelist = set("abcdefghijklmnopqrstuvwxyz ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    for _, row in data.iterrows():
        text = row["text"]
        text = re.sub(
            "([’.,!?()])\"'_", r" \1 ", text
        )  # add padding around punctutation marks
        text = "".join(
            filter(whitelist.__contains__, text)
        )  # remove all non alpha characters

        # needs removing extra data at end, html links, addiitonal spacing
        data_cleaned.append(text)
    return data_cleaned


def data_load(file_name):
    data = pd.read_csv(file_name)
    return data


def str_to_arr(data):
    result = []
    for d in data:
        result.append(literal_eval(d))
    return result


def encode(arr, size):
    res = [0] * size
    for i in arr:
        res[i] = 1
    return res


f = "final_data_cleaned_classification_with_classes_nocumulative.csv"
data = data_load(f)
data_ids = data["Id"].to_list()
tweets = data_cleaner(data["Text"].astype("str"))
d = pd.read_csv("final_data_cleaned_classification_with_classes.csv")
label_col = np.squeeze(d["48 hours"].to_list())
words = []
print(tweets[0])

for tweet in tweets:
    words.extend(prepareSentence(tweet))

distinct_words = set(words)

lower_threshold = 10
upper_threshold = 350
counts = []
final_words = []
print("reached")
from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
X = vect.fit_transform(tweets)
X = X.toarray()
print(len(X[0]))

for word in distinct_words:
    counts.append(words.count(word))
    if words.count(word) > lower_threshold and words.count(word) < upper_threshold:
        final_words.append(word)

print(len(words))
print(len(distinct_words))
print(len(final_words))


def toBOW(sentance, words):
    bag = []
    for word in words:
        bag.append(1) if word in sentance else bag.append(0)
    return bag


inputs = []
outputs = []

for ind in range(len(tweets)):
    sentence = prepareSentence(tweets[ind])
    # create our bag of words array
    bag = toBOW(sentence, final_words)
    inputs.append(bag)
    # Calculate a score, 1 if any engagement, 0 if none
    score = min(label_col[ind], 1)

    outputs.append(score)
print(inputs[2])

# Define and train the network
# nnet = MLPRegressor(activation='relu', alpha=0.0001, hidden_layer_sizes=(int(len(final_words)*0.5),int(len(final_words)*0.25)),solver='adam', max_iter=400)
# nnet.fit(inputs, outputs)
