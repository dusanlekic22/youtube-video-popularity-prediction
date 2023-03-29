# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import one_hot
from keras.utils import pad_sequences
from tensorflow import keras
from keras import layers
from data import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    train, test = split_data(preprocessing_data(import_data()))
    x_train, y_train, x_test, y_test = split_input_output(train, test)

    num_tags = 100  # Number of unique issue tags
    num_words = 10000  # Size of vocabulary obtained when preprocessing text data
    num_pop_levels = 4  # Number of departments for predictions

    title_input = keras.Input(
        shape=(None,), name="title"
    )  # Variable-length sequence of ints
    description_input = keras.Input(shape=(None,), name="description")  # Variable-length sequence of ints
    tags_input = keras.Input(
        shape=(num_tags,), name="tags"
    )  # Binary vectors of size `num_tags`

    # Create keras input for numerical features
    numerical_input = keras.Input(shape=(x_train.iloc[:, np.r_[6:9, 13:15, 17:53]].shape[1],), name="numerical_input")

    # Embed each word in the title into a 64-dimensional vector
    title_features = layers.Embedding(num_words, 64)(title_input)
    # Embed each word in the text into a 64-dimensional vector
    description_features = layers.Embedding(num_words, 64)(description_input)

    # Reduce sequence of embedded words in the title into a single 128-dimensional vector
    title_features = layers.LSTM(128)(title_features)
    # Reduce sequence of embedded words in the description into a single 32-dimensional vector
    description_features = layers.LSTM(32)(description_features)

    # Merge all available features into a single large vector via concatenation
    x = layers.concatenate([title_features, description_features, numerical_input])

    # Stick a department classifier on top of the features
    popularity_pred = layers.Dense(1, name="view_count")(x)

    # Instantiate an end-to-end model predicting both priority and department
    model = keras.Model(
        inputs=[title_input, description_input, numerical_input],
        outputs=[popularity_pred],
    )

    model.compile(
        optimizer=keras.optimizers.RMSprop(1e-3),
        loss=[
            keras.losses.BinaryCrossentropy(from_logits=True),
            keras.losses.CategoricalCrossentropy(from_logits=True),
        ],
        loss_weights=[1.0, 0.2],
        metrics=[keras.metrics.SparseCategoricalAccuracy(), 'AUC', 'accuracy'],
    )

    # Embedding the inputs
    encoded_titles = [one_hot(d, num_words) for d in x_train['title']]
    padded_titles = pad_sequences(encoded_titles, maxlen=6, padding='post')
    encoded_description = [one_hot(d, num_words) for d in x_train['description'].astype(str)]
    padded_description = pad_sequences(encoded_description, maxlen=6, padding='post')
    title_data = padded_titles
    description_data = padded_description
    numerical_data = x_train.iloc[:, np.r_[6:9, 13:15, 17:53]].to_numpy()
    # Dummy target data
    dept_targets = y_train

    model.fit(
        {"title": title_data, "description": description_data,
         "numerical_input": np.asarray(numerical_data).astype(np.float32)},
        {"view_count": dept_targets},
        epochs=15,
        batch_size=32,
    )
    model.save('my_model_2')

    #model = keras.models.load_model("my_model")

    encoded_titles = [one_hot(d, num_words) for d in x_test['title']]
    padded_titles = pad_sequences(encoded_titles, maxlen=6, padding='post')
    encoded_description = [one_hot(d, num_words) for d in x_test['description'].astype(str)]
    padded_description = pad_sequences(encoded_description, maxlen=6, padding='post')
    title_data = padded_titles
    description_data = padded_description
    numerical_data = x_test.iloc[:, np.r_[6:9, 13:15, 17:53]].to_numpy()

    test_scores = model.evaluate([title_data, description_data, numerical_data], y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test sparse accuracy:", test_scores[1])
    print("AUC:", test_scores[2])
    print("Accuracy:", test_scores[3])
    #keras.utils.plot_model(model, "multi_input_and_output_model.png", show_shapes=True)
    prediction = np.round(model.predict([title_data, description_data, numerical_data]))
    #print(y_test['view_count'].reset_index(drop=True), prediction.flatten())
    wrong_predictions = x_test[prediction.flatten() != y_test['view_count']]['title']
    print(wrong_predictions, wrong_predictions.count())

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
