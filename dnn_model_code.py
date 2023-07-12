# -*- coding: utf-8 -*-

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# load dataset
digits = load_digits()
ax = digits.data / 16
ay = digits.target
print(tf.__version__)

sizeOfTest = 0.3
X_train, X_test, y_train, y_test = train_test_split(ax, ay, test_size=sizeOfTest)
print("\nX_train & y_train are train set,\ny_train & y_test are test size. Test size is {:.2%}".format(sizeOfTest))

print(X_train.shape)
len(y_train)


def create_model():
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    return model


model = create_model()
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)  # evaluating before train. just test, no aim
model.fit(X_train, y_train, epochs=10)  # training
model.save('tf_dnn_model')  # saving
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)  # the real testing
print('\nTest accuracy:', test_acc)


# cross validation
def cross_val(ax, ay, split_num):
    ax = np.delete(ax, [0, 4], axis=0)
    ay = np.delete(ay, [0, 4])

    amount = int(len(ax) / 5)
    X_spl = [ax[:amount], ax[amount:2 * amount], ax[2 * amount:3 * amount], ax[3 * amount:4 * amount], ax[4 * amount:]]
    y_spl = [ay[:amount], ay[amount:2 * amount], ay[2 * amount:3 * amount], ay[3 * amount:4 * amount], ay[4 * amount:]]

    X_train_spl = [np.vstack((X_spl[1], X_spl[2], X_spl[3], X_spl[4])),
                   np.vstack((X_spl[0], X_spl[2], X_spl[3], X_spl[4])),
                   np.vstack((X_spl[0], X_spl[1], X_spl[3], X_spl[4])),
                   np.vstack((X_spl[1], X_spl[2], X_spl[0], X_spl[4])),
                   np.vstack((X_spl[1], X_spl[2], X_spl[3], X_spl[0]))]
    y_train_spl = [np.vstack((y_spl[1], y_spl[2], y_spl[3], y_spl[4])),
                   np.vstack((y_spl[0], y_spl[2], y_spl[3], y_spl[4])),
                   np.vstack((y_spl[0], y_spl[1], y_spl[3], y_spl[4])),
                   np.vstack((y_spl[1], y_spl[2], y_spl[0], y_spl[4])),
                   np.vstack((y_spl[1], y_spl[2], y_spl[3], y_spl[0]))]

    acc = []
    for i in range(split_num):
        model2 = create_model()  # important, key, reload model to estimate bias.
        Xcro_train = X_train_spl[i]
        Xcro_test = X_spl[i]
        y_cro_train = y_train_spl[i].flatten()
        y_cro_test = y_spl[i].flatten()

        model2.fit(Xcro_train, y_cro_train, epochs=10, verbose=0)
        test_loss, test_acc = model2.evaluate(Xcro_test, y_cro_test, verbose=0)
        acc.append(test_acc)
    print(acc)
    return np.mean(acc)


print("the cross value accuracy")
print(cross_val(ax, ay, 5))
