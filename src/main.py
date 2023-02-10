# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import one_hot
from keras.utils import pad_sequences
from tensorflow import keras
from tensorflow.keras import layers
from data import *


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    train, test = split_data(import_data())
    x_train, y_train, x_test, y_test = split_input_output(train, test)

    num_tags = 100  # Number of unique issue tags
    num_words = 10000  # Size of vocabulary obtained when preprocessing text data
    num_pop_levels = 4  # Number of departments for predictions

    title_input = keras.Input(
        shape=(None,), name="title"
    )  # Variable-length sequence of ints
    body_input = keras.Input(shape=(None,), name="body")  # Variable-length sequence of ints
    tags_input = keras.Input(
        shape=(num_tags,), name="tags"
    )  # Binary vectors of size `num_tags`

    # Embed each word in the title into a 64-dimensional vector
    title_features = layers.Embedding(num_words, 64)(title_input)
    # Embed each word in the text into a 64-dimensional vector
    body_features = layers.Embedding(num_words, 64)(body_input)

    # Reduce sequence of embedded words in the title into a single 128-dimensional vector
    title_features = layers.LSTM(128)(title_features)
    # Reduce sequence of embedded words in the body into a single 32-dimensional vector
    body_features = layers.LSTM(32)(body_features)

    # Merge all available features into a single large vector via concatenation
    x = layers.concatenate([title_features, body_features, tags_input])

    # Stick a department classifier on top of the features
    popularity_pred = layers.Dense(1, name="view_count")(x)

    # Instantiate an end-to-end model predicting both priority and department
    model = keras.Model(
        inputs=[title_input, body_input, tags_input],
        outputs=[popularity_pred],
    )

    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss=[
            keras.losses.BinaryCrossentropy(from_logits=True),
            keras.losses.CategoricalCrossentropy(from_logits=True),
        ],
        loss_weights=[1.0, 0.2],
        metrics=[keras.metrics.SparseCategoricalAccuracy(),"accuracy"],
    )

    # Dummy input data
    encoded_titles = [one_hot(d, num_words) for d in x_train['title']]
    padded_titles = pad_sequences(encoded_titles, maxlen=6, padding='post')
    encoded_description = [one_hot(d, num_words) for d in x_train['description'].astype(str)]
    padded_description = pad_sequences(encoded_description, maxlen=6, padding='post')
    encoded_tags = [one_hot(d, num_words) for d in x_train['tags']]
    padded_tags = pad_sequences(encoded_tags, maxlen=100, padding='post')
    title_data = padded_titles
    body_data = padded_description
    tags_data = padded_tags
    # Dummy target data
    dept_targets = y_train

    model.fit(
        {"title": title_data, "body": body_data, "tags": tags_data},
        {"view_count": dept_targets},
        epochs=5,
        batch_size=32,
    )

    encoded_titles = [one_hot(d, num_words) for d in x_test['title']]
    padded_titles = pad_sequences(encoded_titles, maxlen=6, padding='post')
    encoded_description = [one_hot(d, num_words) for d in x_test['description'].astype(str)]
    padded_description = pad_sequences(encoded_description, maxlen=6, padding='post')
    encoded_tags = [one_hot(d, num_words) for d in x_test['tags']]
    padded_tags = pad_sequences(encoded_tags, maxlen=100, padding='post')
    title_data = padded_titles
    body_data = padded_description
    tags_data = padded_tags

    test_scores = model.evaluate([title_data, body_data, tags_data], y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
    print("Test sparse accuracy:", test_scores[2])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
