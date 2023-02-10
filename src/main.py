# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from data import *

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    train, test = split_data(import_data())
    x_train, y_train, x_test, y_test = split_input_output(train, test)

    inputs = keras.Input(shape=(128,))
    x = layers.Dense(32, activation="relu")(inputs)
    x = layers.Dense(32, activation="relu")(x)
    outputs = layers.Dense(1)(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.RMSprop(),
        metrics=["accuracy"],
    )
    history = model.fit(x_train, y_train, batch_size=64, epochs=2, validation_split=0.2)

    test_scores = model.evaluate(x_test, y_test, verbose=2)
    print("Test loss:", test_scores[0])
    print("Test accuracy:", test_scores[1])
    model.summary()
    print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
