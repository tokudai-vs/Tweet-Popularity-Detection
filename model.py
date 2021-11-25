# Architecture:

# 1. textual extractor:
#     clean data
#     create embedding vectors for each tweet using keras embedding layer
#     feed embedding vectors to lstm to extract latent textual features

# 2. Retweet count extractor:
#     we already extracted number of retweets in a time slot. File: final_data_cleaned_lstm.csv
#     use one-hot encoder to represent retweets in each time window to  a vector
#     we now have a sequence of vectors for a sequence of time windows corresponding to each tweet.
#     feed each embedding vectors to single gru to exract latent features.
#     deploy attention layer after gru

# We're not considering the 3rd module since we don't have the required data. Future possiblity of including follower count

# 3. Final module:
#     concatenate lstm and gru


import contractions
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Embedding, LSTM, Dropout
from tensorflow.keras.models import Sequential
from keras.preprocessing.text import one_hot, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import concatenate, Dense
from tensorflow.keras.layers import GRU, Dense, Attention, Multiply
from tensorflow.keras.layers import Embedding, LSTM, Input
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras import Model
import json
import io
import pandas as pd
from sklearn.utils import class_weight
import numpy as np
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import random

np.random.seed(0)
random.seed(9)
tf.random.set_seed(123)
from numpy import array
import re
import nltk
from ast import literal_eval
from sklearn.metrics import f1_score, log_loss, precision_recall_fscore_support

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
    return max(results) + 1


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


def normalize(data):  # min-max normalization of explicit features
    min_data = min(data)
    max_data = max(data)
    diff = max_data - min_data
    res = [round((float(i) - min_data) / diff, 7) for i in data]
    return res


def data_cleaner(data):

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
        text = contractions.fix(text)  # fix contractions like don't to do not
        text = "".join(
            filter(whitelist.__contains__, text)
        )  # remove all non alpha characters
        # needs removing extra data at end, html links, addiitonal spacing
        text = text[
            : text.find("Name text dtype object")
        ]  # Caused due to dtype in dataset
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


def save_model(model, model_path):
    model_json = model.to_json()
    with open(model_path, "w") as json_file:
        json_file.write(model_json)


def train(
    model,
    X_train,
    y_train,
    X_val,
    y_val,
    checkpoint_path="model.hdf5",
    epochs=25,
    steps_per_epoch=50,
    batch_size=32,
    class_weights=None,
    fit_verbose=1,
    print_summary=True,
):
    if print_summary:
        print(model.summary())
    model.fit(
        X_train,
        y_train,
        # this is bad practice using test data for validation, in a real case would use a seperate validation set
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        steps_per_epoch=steps_per_epoch,
        class_weight=class_weights,
        # saves the most accurate model, usually you would save the one with the lowest loss
        callbacks=[
            ModelCheckpoint(
                checkpoint_path,
                monitor="val_loss",
                verbose=1,
                save_best_only=True,
                mode="min",
            ),
            EarlyStopping(patience=2),
        ],
        verbose=fit_verbose,
    )
    return model


def test(model, X_test, y_test, checkpoint_path):

    print("Loading Best Model...")
    model.load_weights(checkpoint_path)
    predictions = model.predict(X_test, verbose=1)
    y_test_arr = np.asarray(y_test)
    print("Test Loss:", log_loss(y_test_arr, predictions))
    print("Test Accuracy", predictions.argmax(axis=1) == y_test_arr.argmax(axis=1))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test_arr.argmax(axis=1), predictions.argmax(axis=1), average="weighted"
    )
    print("Precision: {0}, Recall: {1},F1 Score: {2}".format(precision, recall, f1))

    return model  # returns best performing model


# read data

f = "dataset_final.csv"
data = data_load(f)
data_ids = data["Id"].to_list()

data_text = data_cleaner(data["Text"].astype("str"))
f = "explicit_features.csv"
d = data_load(f)

explicit_features = [list(row) for row in d.values]

data_retweets = str_to_arr(data["Retweets"].to_list())
data_retweets = [
    data[:6] for data in data_retweets
]  # with time_window = 10min and observation time = 1 hour

vocab_size = vocab_count(data)
retweet_class_size = retweet_class_count(data_retweets)

tokenizer = Tokenizer(lower=False)
tokenizer.fit_on_texts(data_text)

# Uncomment for saving the tokenizer
# tokenizer_json = tokenizer.to_json()
# with io.open('tokenizer-store.json', 'w', encoding='utf-8') as f:
#     f.write(json.dumps(tokenizer_json, ensure_ascii=False))

encoded_text = tokenizer.texts_to_sequences(data_text)
max_pad_length = 25  # number of words to be padded
padded_text = pad_sequences(
    encoded_text, maxlen=max_pad_length, padding="post"
)  # done so that all sentences are of same size

encoded_retweet_counts = np.array(
    [to_categorical(d, num_classes=retweet_class_size) for d in data_retweets]
)

labels = []
d = pd.read_csv("final_data_cleaned_classification_with_classes.csv")
label_col = np.squeeze(d["48 hours"].to_list())

thres = 700  # can be varied, class definition is based on it
for t in label_col:
    if t < thres:
        labels.append(0)
    else:
        labels.append(1)
labels = to_categorical(labels, 2)

# LSTM model
input_shape_texts = padded_text.shape[1]
input_textExt = Input(shape=input_shape_texts)
tEl1 = Embedding(vocab_size, 128, input_length=input_shape_texts)(input_textExt)
tEl2 = LSTM(units=256, activation="relu", dropout=0.4)(tEl1)

# GRU model
input_shape = encoded_retweet_counts.shape
input_shape_rCE = (
    input_shape[1],
    input_shape[2],
)
input_rCE = Input(shape=input_shape_rCE)
rCEl1 = GRU(units=256, dropout=0.1, activation="relu")(input_rCE)
attention_probs = Dense(
    units=256,
    activity_regularizer=regularizers.l2(0.01),
    activation="relu",
    name="attention_probs",
)(rCEl1)
attention_mul = Multiply(name="attention_mul")([rCEl1, attention_probs])
rcEl2 = Dense(256, activity_regularizer=regularizers.l2(0.01), activation="sigmoid")(
    attention_mul
)

# Explicit features
input_shape_ef = len(explicit_features[0])
input_ef = Input(shape=input_shape_ef)

# Final model
merged1 = concatenate([tEl2, rcEl2])
fl1 = Dense(256, activity_regularizer=regularizers.l2(0.01), activation="relu")(merged1)
fl2 = Dense(256, activity_regularizer=regularizers.l2(0.01), activation="sigmoid")(fl1)
merged2 = concatenate([fl2, input_ef])
fl3 = Dense(2, activity_regularizer=regularizers.l2(0.01), activation="softmax")(
    merged2
)
model = Model(inputs=[input_textExt, input_rCE, input_ef], outputs=[fl3])

# Uncomment to save the model
# save_model(model,"./model.json")

# randomly split of data into train,validation and test data
x1_train = []
x2_train = []
x3_train = []
y_train = []
x1_val = []
x2_val = []
x3_val = []
y_val = []
x1_test = []
x2_test = []
x3_test = []
y_test = []

texts = []
retweet_count = []
for ind in range(len(padded_text)):
    r = random.random()
    if r < 0.4:
        x1_train.append(padded_text[ind])
        x2_train.append(encoded_retweet_counts[ind])
        x3_train.append(explicit_features[ind])
        y_train.append(labels[ind])

    elif 0.4 <= r < 0.6:
        x1_val.append(padded_text[ind])
        x2_val.append(encoded_retweet_counts[ind])
        x3_val.append(explicit_features[ind])
        y_val.append(labels[ind])

    else:
        x1_test.append(padded_text[ind])
        x2_test.append(encoded_retweet_counts[ind])
        x3_test.append(explicit_features[ind])
        y_test.append(labels[ind])
        texts.append(data_text[ind])
        retweet_count.append(label_col[ind])

print("Converting to tensors")

x1_train = tf.convert_to_tensor(x1_train)
x2_train = tf.convert_to_tensor(x2_train)
x3_train = tf.convert_to_tensor(x3_train)
y_train = tf.convert_to_tensor(y_train)
x1_val = tf.convert_to_tensor(x1_val)
x2_val = tf.convert_to_tensor(x2_val)
x3_val = tf.convert_to_tensor(x3_val)
y_val = tf.convert_to_tensor(y_val)
x1_test = tf.convert_to_tensor(x1_test)
x2_test = tf.convert_to_tensor(x2_test)
x3_test = tf.convert_to_tensor(x3_test)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])

model = train(
    model,
    checkpoint_path="./model_1-classify.h5",
    X_train=[x1_train, x2_train, x3_train],
    y_train=y_train,
    X_val=[x1_val, x2_val, x3_val],
    y_val=y_val,
    epochs=15,
)
model = test(
    model,
    checkpoint_path="./model_1-classify.h5",
    X_test=[x1_test, x2_test, x3_test],
    y_test=y_test,
)
