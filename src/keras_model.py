# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from collections import Counter

import numpy as np
import tensorflow as tf
from keras.layers import TextVectorization
from keras.preprocessing.text import one_hot
from keras.utils import pad_sequences
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_absolute_error
from tensorflow import keras
from keras import layers
from data import *

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train, test = split_data(preprocessing_data(import_data()))
    x_train, y_train, x_test, y_test = split_input_output(train, test)

    num_tags = 100  # Number of unique issue tags
    num_words = 20000  # Size of vocabulary obtained when preprocessing text data
    sequence_length = 500

    vectorize_title_layer = TextVectorization(
        max_tokens=20000,
        output_mode="int",
        output_sequence_length=500,
    )

    vectorize_description_layer = TextVectorization(
        max_tokens=num_words,
        output_mode="int",
        output_sequence_length=sequence_length,
    )

    vectorize_tags_layer = TextVectorization(
        max_tokens=num_words,
        output_mode="int",
        output_sequence_length=300,
    )

    title_input = keras.Input(shape=(1,), dtype=tf.string, name='title')
    # Variable-length sequence of ints
    description_input = keras.Input(shape=(1,), dtype=tf.string, name='description')
    # Variable-length sequence of ints
    tags_input = keras.Input(shape=(1,), dtype=tf.string, name="tags")
    # Binary vectors of size `num_tags`

    # Create keras input for numerical features
    numerical_input = keras.Input(shape=(x_train.drop(['title', 'description', 'tags'], axis=1).shape[1],),
                                  name="numerical_input")
    # create layers for numerical features
    numerical_features = layers.Dense(64, activation="relu")(numerical_input)
    numerical_features = layers.Dense(64, activation="relu")(numerical_features)
    # Let's call `adapt`:
    vectorize_title_layer.adapt(x_train['title'].values)
    x = vectorize_title_layer(title_input)
    # Embed each word in the title into a 64-dimensional vector
    title_features = layers.Embedding(num_words+1, 64)(x)
    # Embed each word in the text into a 64-dimensional vector
    # Let's call `adapt`:
    vectorize_description_layer.adapt(x_train['description'].values)
    x = vectorize_description_layer(description_input)
    description_features = layers.Embedding(num_words+1, 64)(x)

    #vectorize_tags_layer.adapt(x_train['tags'].values)
    x = vectorize_tags_layer(tags_input)
    # Embed each tag into a 64-dimensional vector
    tags_features = layers.Embedding(num_words+1, 64)(x)

    # Reduce sequence of embedded words in the title into a single 128-dimensional vector
    title_features = layers.LSTM(32)(title_features)
    # Reduce sequence of embedded words in the description into a single 32-dimensional vector
    description_features = layers.LSTM(32)(description_features)
    # Reduce sequence of embedded words in the tags into a single 32-dimensional vector
    tags_features = layers.LSTM(32)(tags_features)

    # Merge all available features into a single large vector via concatenation
    x = layers.concatenate([title_features, description_features, numerical_features])

    # Stick a department classifier on top of the features
    popularity_pred = layers.Dense(1, activation='sigmoid', name="view_count")(x)

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
        loss_weights=[2.0, 0.2],
        metrics=['mean_absolute_error', 'AUC', 'accuracy'],
    )

    title_data = x_train['title']
    description_data = x_train['description']
    #tags_data = x_train['tags']

    # encoded_titles = [one_hot(d, num_words) for d in x_train['title']]
    # padded_titles = pad_sequences(encoded_titles, maxlen=6, padding='post')
    # encoded_description = [one_hot(d, num_words) for d in x_train['description'].astype(str)]
    # padded_description = pad_sequences(encoded_description, maxlen=6, padding='post')
    # title_data = padded_titles
    # description_data = padded_description
    numerical_data = x_train.drop(['title', 'description', 'tags'], axis=1).to_numpy()

    model.fit(
        {"title": title_data, "description":description_data,
         "numerical_input": np.asarray(numerical_data).astype(np.float32)},
        {"view_count": y_train},
        epochs=6,
        batch_size=32,
    )
    model.save('keras_models/my_model_10')

    #model = keras.models.load_model("keras_models/my_model_10")

    title_data = x_test['title']
    description_data = x_test['description']
    #tags_data = x_test['tags']

    # encoded_titles = [one_hot(d, num_words) for d in x_test['title']]
    # padded_titles = pad_sequences(encoded_titles, maxlen=6, padding='post')
    # encoded_description = [one_hot(d, num_words) for d in x_test['description'].astype(str)]
    # padded_description = pad_sequences(encoded_description, maxlen=6, padding='post')
    # title_data = padded_titles
    # description_data = padded_description
    numerical_data = x_test.drop(['title', 'description', 'tags'], axis=1).to_numpy()

    test_scores = model.evaluate([title_data, description_data, numerical_data], y_test, verbose=2)

    keras.utils.plot_model(model, "keras_models/multi_input_and_output_model.png", show_shapes=True)
    prediction = np.round(model.predict([numerical_data]))
    #evaluate the model with accuracy,f1,auc score
    accuracy = accuracy_score(y_test, prediction)
    auc_roc = roc_auc_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    micro_f1 = f1_score(y_test, prediction, average='micro')

    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("ROC: %.2f%%" % (auc_roc * 100.0))
    print("MAE: ", mean_absolute_error(y_test, prediction)*100)
    print("F1: %.2f%%" % (f1 * 100.0))
    wrong_predictions = x_test[abs(prediction.flatten() - y_test['view_count']) > 0.5]
    print(wrong_predictions.columns)
    wrong_predictions[['likes', 'dislikes', 'comment_count', 'channel_view_count', 'channel_subscribe_count']].hist()
    plt.show()
    print(wrong_predictions, wrong_predictions.count(), x_test.count())
    #find the most recurring words in the title and description in the wrong predictions
    title_words = []
    description_words = []
    for index, row in wrong_predictions.iterrows():
        title_words.extend(row['title'].split())
        description_words.extend(row['description'].split())
    title_words = Counter(title_words)
    description_words = Counter(description_words)
    print(title_words.most_common(20), description_words.most_common(20))



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
