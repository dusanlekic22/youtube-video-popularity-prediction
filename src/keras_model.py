# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
from collections import Counter
from keras.layers import TextVectorization
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_absolute_error
from tensorflow import keras
from keras import layers
from data import *
from sentence_transformers import SentenceTransformer


def encode_sentence(sent):
    return model.encode(sent)


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

    title_input = keras.Input(shape=(1,), dtype=tf.string, name='title')
    # Binary vectors of size `num_tags
    thumbnail_input = keras.Input(shape=(120, 90, 3), name="thumbnail")

    # Create keras input for numerical features
    numerical_input = keras.Input(shape=(x_train.drop(['title', 'description', 'tags', 'video_id'], axis=1).shape[1],),
                                  name="numerical_input")
    # create layers for numerical features
    numerical_features = layers.Dense(64, activation="relu")(numerical_input)
    numerical_features = layers.Dense(64, activation="relu")(numerical_features)
    # Let's call `adapt`:
    vectorize_title_layer.adapt(x_train['title'].values)
    x = vectorize_title_layer(title_input)
    # Embed each word in the title into a 64-dimensional vector
    title_features = layers.Embedding(num_words+1, 64)(x)

    # Apply the encoding function to each sentence in the input tensor
    # Reduce sequence of embedded words in the title into a single 128-dimensional vector
    title_features = layers.LSTM(32)(title_features)

    x = layers.Conv2D(32, 3, activation="relu")(thumbnail_input)
    x = layers.Conv2D(64, 3, activation="relu")(x)
    block_1_output = layers.MaxPooling2D(3)(x)

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_1_output)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    block_2_output = layers.add([x, block_1_output])

    x = layers.Conv2D(64, 3, activation="relu", padding="same")(block_2_output)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    block_3_output = layers.add([x, block_2_output])

    x = layers.Conv2D(64, 3, activation="relu")(block_3_output)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    thumbnail_features = layers.Dense(10)(x)

    # thumbnail_features = keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=(120, 90, 3))\
    #     (thumbnail_input)

    # x = layers.Conv2D(64, 3, activation="relu")(thumbnail_features)
    # x = layers.GlobalAveragePooling2D()(x)
    # x = layers.Dense(256, activation="relu")(x)
    # x = layers.Dropout(0.5)(x)
    # thumbnail_features = layers.Dense(10)(x)

    # Merge all available features into a single large vector via concatenation
    x = layers.concatenate([title_features, numerical_features, thumbnail_features])

    # Stick a department classifier on top of the features
    popularity_pred = layers.Dense(1, activation='sigmoid', name="view_count")(x)

    # Instantiate an end-to-end model predicting both priority and department
    model = keras.Model(
        inputs=[title_input, numerical_input, thumbnail_input],
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

    #select only the title column
    title_data = x_train['title']
    # select everything but the title column
    thumbnail_data = load_images(x_train)/255
    #tags_data = x_train['tags']
    # encoded_titles = [one_hot(d, num_words) for d in x_train['title']]
    # padded_titles = pad_sequences(encoded_titles, maxlen=6, padding='post')
    # encoded_description = [one_hot(d, num_words) for d in x_train['description'].astype(str)]
    # padded_description = pad_sequences(encoded_description, maxlen=6, padding='post')
    # title_data = padded_titles
    # description_data = padded_description
    numerical_data = x_train.drop(['video_id', 'title', 'description', 'tags'], axis=1).to_numpy()
    print(thumbnail_data.shape, len(numerical_data), len(y_train))

    model.fit(
        {"title": title_data,
         "numerical_input": numerical_data,
         "thumbnail": thumbnail_data},
        {"view_count": y_train},
        epochs=6,
        batch_size=32,
    )
    model.save('keras_models/my_model_11')

    #model = keras.models.load_model("keras_models/my_model_10")

    title_data = x_test['title']
    #tags_data = x_test['tags']
    thumbnail_data = load_images(x_test)/255
    # encoded_titles = [one_hot(d, num_words) for d in x_test['title']]
    # padded_titles = pad_sequences(encoded_titles, maxlen=6, padding='post')
    # encoded_description = [one_hot(d, num_words) for d in x_test['description'].astype(str)]
    # padded_description = pad_sequences(encoded_description, maxlen=6, padding='post')
    # title_data = padded_titles
    # description_data = padded_description
    numerical_data = x_test.drop(['video_id', 'title', 'description', 'tags'], axis=1).to_numpy()

    test_scores = model.evaluate([title_data, numerical_data, thumbnail_data], y_test, verbose=2)

    keras.utils.plot_model(model, "keras_models/multi_input_and_output_model.png", show_shapes=True)
    prediction = np.round(model.predict([title_data, numerical_data, thumbnail_data]))
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
