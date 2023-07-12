# -*- coding: utf-8 -*-

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, models

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

# load dataset
digits = load_digits()
ax = digits.data
ax = ax.reshape((len(ax), 8, 8))
ay = digits.target
print(tf.__version__)

ax = ax.reshape(ax.shape[0], 8, 8, 1).astype('float32')
ax = tf.keras.utils.normalize(ax, axis=1)

sizeOfTest = 0.3
X_train, X_test, y_train, y_test = train_test_split(ax, ay, test_size=sizeOfTest)
print("\nX_train & y_train are train set,\ny_train & y_test are test size. Test size is {:.2%}".format(sizeOfTest))

#  view the shape
X_train.shape


def create_model():
    model = keras.models.Sequential([
        layers.Conv2D(filters=16, kernel_size=(2, 2), padding='same',
                      input_shape=(8, 8, 1), activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Conv2D(filters=32, kernel_size=(2, 2), padding='same',
                      activation='relu'),
        layers.MaxPooling2D(pool_size=(2, 2)),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(16, activation='relu'),
        layers.Dense(10, activation='softmax')
    ])

    # print model structure
    print(model.summary())

    # compile
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


# begin train and save model
model = create_model()
model.fit(x=X_train, y=y_train, epochs=10, verbose=1)
model.save('tf_cnn_model')
# evaluate by test set
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print('\nTest accuracy:', test_acc)


#  cross validation
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


