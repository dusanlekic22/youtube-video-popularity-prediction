# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import one_hot
from keras.utils import pad_sequences
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
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
    numerical_input = keras.Input(shape=(x_train.drop(['title', 'description'], axis=1).shape[1],), name="numerical_input")

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

    # Embedding the inputs
    encoded_titles = [one_hot(d, num_words) for d in x_train['title']]
    padded_titles = pad_sequences(encoded_titles, maxlen=6, padding='post')
    encoded_description = [one_hot(d, num_words) for d in x_train['description'].astype(str)]
    padded_description = pad_sequences(encoded_description, maxlen=6, padding='post')
    title_data = padded_titles
    description_data = padded_description
    numerical_data = x_train.drop(['title', 'description'], axis=1).to_numpy()
    # Dummy target data
    dept_targets = y_train

    model.fit(
        {"title": title_data, "description": description_data,
         "numerical_input": np.asarray(numerical_data).astype(np.float32)},
        {"view_count": dept_targets},
        epochs=6,
        batch_size=32,
    )
    model.save('keras_models/my_model_9')

    #model = keras.models.load_model("keras_models/my_model")

    encoded_titles = [one_hot(d, num_words) for d in x_test['title']]
    padded_titles = pad_sequences(encoded_titles, maxlen=6, padding='post')
    encoded_description = [one_hot(d, num_words) for d in x_test['description'].astype(str)]
    padded_description = pad_sequences(encoded_description, maxlen=6, padding='post')
    title_data = padded_titles
    description_data = padded_description
    numerical_data = x_test.drop(['title', 'description'], axis=1).to_numpy()

    test_scores = model.evaluate([title_data, description_data, numerical_data], y_test, verbose=2)

    #keras.utils.plot_model(model, "keras_models/multi_input_and_output_model.png", show_shapes=True)
    prediction = np.round(model.predict([title_data, description_data, numerical_data]))
    #evaluate the model with accuracy,f1,auc score
    accuracy = accuracy_score(y_test, prediction)
    auc_roc = roc_auc_score(y_test, prediction)
    f1 = f1_score(y_test, prediction)
    micro_f1 = f1_score(y_test, prediction, average='micro')
    # bst = xgb.Booster({'nthread': 4})  # init model
    # bst.load_model('0001.model')  # load data
    print("Accuracy: %.2f%%" % (accuracy * 100.0))
    print("ROC: %.2f%%" % (auc_roc * 100.0))
    #print("F1: %.2f%%" % (f1 * 100.0))
    print("Micro F1: %.2f%%" % (micro_f1 * 100.0))
    # wrong_predictions = x_test[abs(prediction.flatten() - y_test['view_count']) < 0.5]
    # print(wrong_predictions.columns)
    # wrong_predictions.hist()
    # plt.show()
    # print(wrong_predictions, wrong_predictions.count(), x_test.count())



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
