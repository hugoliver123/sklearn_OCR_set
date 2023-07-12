# -*- coding: utf-8 -*-
# TensorFlow and tf.keras
import joblib
import sklearn
import tensorflow as tf

# Helper libraries
import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


# distance and knnSelf are source code for my self knn
# distance aims to calculate Euclid distance
def distance(d1, d2):
    dis_res = 0

    for key in range(0, 64):
        dis_res += ((d1[key]) - (d2[key])) ** 2

    return dis_res ** 0.5


# knnSelf is the gate, also the body of the algorithm
def knnSelf(data, k, X_train_sub, y_train_sub):
    res_list = []
    proba_list = []
    for index in data:
        num = 0
        res = [
            {"result": y_train_sub[num], "distance": distance(index, X_train_sub[num])}
            for num in range(0, len(X_train_sub))
        ]
        # 2 Sort, collect k item
        res = sorted(res, key=lambda item: item['distance'])
        res2 = res[0:k]
        # 3 average
        result = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0, 7: 0, 8: 0, 9: 0}
        sum_distance = 0
        for r in res2:
            sum_distance += r['distance']
        for r in res2:
            result[r['result']] += 1 - r['distance'] / sum_distance
        max_avg = 0

        # allocating
        temp = 0
        temp_arr = []
        for i in range(0, 10):
            temp_arr.append(result[i] / 10)
            if result[i] > max_avg:
                max_avg = result[i]
                temp = i
        proba_list.append(temp_arr)
        res_list.append(temp)
    return res_list, proba_list  # return two list. one is result list, another is probably


def load_myModel(model_path):
    # method for load model. knn and tf are different load methods.
    if model_path.startswith('tf'):
        model = tf.keras.models.load_model(model_path)
    else:
        model = joblib.load(model_path)
    return model


# Cross validation
def cross_val(ax_sub, ay_sub, model_path):
    # if the dataset cannot divide by 5, randomly removing element to meet the criteria.
    if len(ax_sub) % 5 != 0:
        ran = random.randint(0, len(ax_sub) - 4)
        ax_sub = np.delete(ax_sub, [ran, ran + len(ax_sub) % 5 + 1], axis=0)
        ay_sub = np.delete(ay_sub, [ran, ran + len(ax_sub) % 5 + 1])

    # split dataset and reorder them for cross validation
    amount = int(len(ax_sub) / 5)
    X_spl = [ax_sub[:amount], ax_sub[amount:2 * amount], ax_sub[2 * amount:3 * amount], ax_sub[3 * amount:4 * amount],
             ax_sub[4 * amount:]]
    y_spl = [ay_sub[:amount], ay_sub[amount:2 * amount], ay_sub[2 * amount:3 * amount], ay_sub[3 * amount:4 * amount],
             ay_sub[4 * amount:]]

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

    # calculate each acc value, and build a list with 5 elements
    acc = []  # list of accuracy of each crossing validation.
    pointer = 0  # pointer is current validating index, total is 5 here.
    for i in range(5):
        Xcro_train = X_train_spl[i]  # X for crossing training sets.
        Xcro_test = X_spl[i]  # X for crossing testing sets.
        y_cro_train = y_train_spl[i].flatten()  # y for crossing training sets.
        y_cro_test = y_spl[i].flatten()  # y for crossing testing sets.
        pointer += 1
        print("cross validating! current: %d, total:5" % pointer)
        acc.append(cor_train_test(Xcro_train, y_cro_train, Xcro_test, y_cro_test, model_path))
    print("detail accuracy:", acc)
    print("average:", np.mean(acc))
    return np.mean(acc)  # return list of accuracy, capacity is 5, here.


def cor_train_test(X_cro_train, y_cro_train, X_cro_test, y_cro_test, model_path):
    if model_path.startswith('nil'):  # to keep uniformity, myself knn without model need to add path 'nil'
        # get predict result
        knn_predictions_sub, knn_pro_sub = knnSelf(X_cro_test, 5, X_cro_train, y_cro_train)
        # calculate and return accuracy
        corr = 0
        for i in range(0, len(y_cro_test)):
            if knn_predictions_sub[i] == y_cro_test[i]:
                corr += 1
        return corr / len(y_cro_test)
    else:  # here is neural network and knn from sklearn
        # load model
        model = load_myModel(model_path)
        # training model with train sets
        if model_path.startswith('tf'):  # neural network
            model.fit(X_cro_train, y_cro_train, epochs=network_epochs, verbose=0)
        else:  # knn from sklearn
            model.fit(X_cro_train, y_cro_train)
        # test model with test sets
        if model_path.startswith('tf'):
            test_loss, test_acc = model.evaluate(X_cro_test, y_cro_test, verbose=1)
        else:
            test_acc = model.score(X_cro_test, y_cro_test)
        # return the acc value of cross validation.
        return test_acc


# confusion matrix
def cul_confusion_matrix(model_path, X_train_sub, y_train_sub, X_test_sub, y_test_sub):
    model = load_myModel(model_path)  # call load model function

    if model_path.startswith('tf'):  # neural network
        #  model.fit(X_train_sub, y_train_sub, epochs=network_epochs, verbose=0)
        print("no need to train")
    else:  # knn from sklearn, knn model not save train result need to train in here.
        model.fit(X_train_sub, y_train_sub)

    C_matrix = np.zeros((10, 10), dtype=np.int)  # defending list of Crossing matrix

    predictions = model.predict(X_test_sub)  # get model predict result
    for i in range(0, len(X_test_sub)):
        if model_path.startswith('tf'):
            num_predict = np.argmax(predictions[i])  # tensorflow neural network. the predictions are probably.
        else:
            num_predict = predictions[i]  # KNN from sklearn. the predictions are specific and clear class
        C_matrix[num_predict][y_test_sub[i]] += 1  # allocate result into confusion matrix
    print(C_matrix)
    return C_matrix


#  because of different arrangement of myself knn, there is a private method.
#  the structure and realizing ways are same, please reference following confusion_matrix()
def self_cul_matrix(prediction_sub, X_test_sub, y_test_sub):
    C_matrix = np.zeros((10, 10), dtype=np.int)
    for i in range(0, len(X_test_sub)):
        num_predict = prediction_sub[i]
        num_fact = y_test_sub[i]
        C_matrix[num_predict][num_fact] += 1
    print(C_matrix)
    plt.matshow(C_matrix, cmap='PuBu', aspect='auto')
    plt.colorbar()
    plt.title("confusion matrix; model: self_knn")
    for i in range(10):
        for j in range(10):
            plt.text(x=j, y=i, s=C_matrix[i][j])
    plt.show()
    return C_matrix


# draw confusion matrix via plt
def confusion_matrix(model_path, X_train_sub, y_train_sub, X_test_sub, y_test_sub):
    matrix = cul_confusion_matrix(model_path, X_train_sub, y_train_sub, X_test_sub, y_test_sub)
    plt.matshow(matrix, cmap='PuBu', aspect='auto')
    plt.colorbar()
    plt.title("confusion matrix; model: %s" % model_path)
    for i in range(10):
        for j in range(10):
            plt.text(x=j, y=i, s=matrix[i][j])
    plt.show()
    return matrix


# calculate tpr and fpr.
def get_tfpr(X_test_sub, y_test_sub, threshold, model, model_path):
    tpr = []
    fpr = []
    matrix_set = []  # each element in the list is confusion matrix with a certain class. capacity 10.
    for i in range(0, 10):
        matrix_set.append(np.zeros((2, 2), dtype=np.int))  # initializing 10 sub_matrix

    if model_path.startswith('tf'):  # neural network, tensorFlow
        prediction = model.predict(X_test_sub)
    else:  # knn sklearn
        prediction = model.predict_proba(X_test_sub)

    for class_num in range(0, 10):
        for index in range(0, len(X_test_sub)):
            pointer = y_test_sub[index]
            if pointer == class_num:
                if prediction[index][class_num] > threshold:
                    (matrix_set[class_num])[1][1] += 1.  # (predict positive; fact true) TP
                else:
                    (matrix_set[class_num])[1][0] += 1  # (predict negative; fact true)  FN
            else:
                if prediction[index][class_num] > threshold:
                    (matrix_set[class_num])[0][1] += 1  # (predict positive; fact false) FP
                else:
                    (matrix_set[class_num])[0][0] += 1  # (predict negative; fact false) TN
        tpr.append((matrix_set[class_num])[1][1] / ((matrix_set[class_num])[1][0] + (matrix_set[class_num])[1][1]))
        fpr.append((matrix_set[class_num])[0][1] / ((matrix_set[class_num])[0][1] + (matrix_set[class_num])[0][0]))
    return tpr, fpr  # tpr and fpr are lists in given threshold.


#  because of different arrangement of myself knn, there is a private method.
#  the structure and realizing ways are same, please reference above get_tfpr()
def self_get_tfpr(X_test_sub, y_test_sub, threshold, prediction):  # here prediction is prediction already ran.
    tpr = []
    fpr = []
    matrix_set = []
    for i in range(0, 10):
        matrix_set.append(np.zeros((2, 2), dtype=np.int))
    for class_num in range(0, 10):
        for index in range(0, len(X_test_sub)):
            pointer = y_test_sub[index]
            if pointer == class_num:
                if prediction[index][class_num] > threshold:
                    (matrix_set[class_num])[1][1] += 1.  # (predict positive; fact true) TP
                else:
                    (matrix_set[class_num])[1][0] += 1  # (predict negative; fact true)  FN
            else:
                if prediction[index][class_num] > threshold:
                    (matrix_set[class_num])[0][1] += 1  # (predict positive; fact false) FP
                else:
                    (matrix_set[class_num])[0][0] += 1  # (predict negative; fact false) TN
        tpr.append((matrix_set[class_num])[1][1] / ((matrix_set[class_num])[1][0] + (matrix_set[class_num])[1][1]))
        fpr.append((matrix_set[class_num])[0][1] / ((matrix_set[class_num])[0][1] + (matrix_set[class_num])[0][0]))
    return tpr, fpr  # tpr and fpr are lists in given threshold.


#  get resource which drawing ROC needed.
def get_roc_resource(model_path, X_train_sub, y_train_sub, X_test_sub, y_test_sub):
    model = load_myModel(model_path)
    if model_path.startswith('tf'):
        #  model.fit(X_train_sub, y_train_sub, epochs=network_epochs, verbose=0)
        print("no need to train")
    else:  # knn from sklearn, knn model not save train result need to train in here.
        model.fit(X_train_sub, y_train_sub)

    tpr = []  # tpr[threshold][class_num]
    fpr = []  # fpr[threshold][class_num]

    for threshold in range(0, 10):  # my ROC need 10 different threshold.
        # this loop is test each and create a result list
        tpr_temp, fpr_temp = get_tfpr(X_test_sub, y_test_sub, threshold / 10, model, model_path)
        tpr.append(tpr_temp)
        fpr.append(fpr_temp)
    res = [tpr, fpr]
    return res


#  because of different arrangement of myself knn, there is a private method.
#  the structure and realizing ways are similar, please reference above get_roc_resource()
def self_get_roc_resource(X_test_sub, y_test_sub, pro_sub):
    tpr = []  # tpr[threshold][class_num]
    fpr = []  # fpr[threshold][class_num]

    for threshold in range(0, 10):
        tpr_temp, fpr_temp = self_get_tfpr(X_test_sub, y_test_sub, threshold / 10, pro_sub)
        tpr.append(tpr_temp)
        fpr.append(fpr_temp)
    res = [tpr, fpr]
    return res


#  draw ROC curve via plt.
def draw_roc(model_path, X_train_sub, y_train_sub, X_test_sub, y_test_sub):
    plt.figure()
    resource_roc = get_roc_resource(model_path, X_train_sub, y_train_sub, X_test_sub,
                                    y_test_sub)  # resource[0:tpr; 1:fpr][threshold][class_num]
    class_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colour = ['brown', 'peru', 'gold', 'lime', 'cyan', 'slategray', 'blue', 'darkviolet', 'magenta', 'pink']
    lw = 3
    for j in range(0, 10):
        roc_tpr = []
        roc_fpr = []
        for i in range(0, 10):
            roc_tpr.append(resource_roc[0][i][j])
            roc_fpr.append(resource_roc[1][i][j])
        plt.plot(roc_fpr, roc_tpr, color=colour[j],
                 lw=lw, label='class: %d' % class_index[j])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic; model: %s' % model_path)
    plt.legend(loc="lower right")
    plt.show()


#  because of different arrangement of myself knn, there is a private method.
#  the structure and realizing ways are similar, please reference above draw_roc()
def self_draw_roc(X_test_sub, y_test_sub, pro_sub):
    plt.figure()
    resource_roc = self_get_roc_resource(X_test_sub, y_test_sub,
                                         pro_sub)  # resource[0:tpr; 1:fpr][threshold][class_num]
    class_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    colour = ['brown', 'peru', 'gold', 'lime', 'cyan', 'slategray', 'blue', 'darkviolet', 'magenta', 'pink']
    lw = 3
    for j in range(0, 10):
        roc_tpr = []
        roc_fpr = []
        for i in range(0, 10):
            roc_tpr.append(resource_roc[0][i][j])
            roc_fpr.append(resource_roc[1][i][j])
        plt.plot(roc_fpr, roc_tpr, color=colour[j],
                 lw=lw, label='class: %d' % class_index[j])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic; model: self_knn')
    plt.legend(loc="lower right")
    plt.show()


def test_knn(ax_sub, ay_sub, X_train_sub, y_train_sub, X_test_sub, y_test_sub):
    print("\n\n********KNN********")
    # model path:
    model_path = 'knn.joblib'
    # cross validation
    print("\n________The cross validation:________")
    cross_val(ax_sub, ay_sub, model_path)
    #  confusion matrix
    print("\n________The confusion matrix________")
    confusion_matrix(model_path, X_train_sub, y_train_sub, X_test_sub, y_test_sub)
    #  ROC
    draw_roc(model_path, X_train_sub, y_train_sub, X_test_sub, y_test_sub)


def test_cnn(ax_sub, ay_sub, X_train_sub, y_train_sub, X_test_sub, y_test_sub):
    print("\n\n********CNN********")
    # model path:
    model_path = 'tf_cnn_model'  # tf is a keyword, please make sure begin with.

    # reshape dataset for cnn using
    ax_sub = ax_sub.reshape(ax_sub.shape[0], 8, 8, 1).astype('float32')
    ax_sub = tf.keras.utils.normalize(ax_sub, axis=1)

    X_train_sub = X_train_sub.reshape(X_train_sub.shape[0], 8, 8, 1).astype('float32')
    X_train_sub = tf.keras.utils.normalize(X_train_sub, axis=1)

    X_test_sub = X_test_sub.reshape(X_test_sub.shape[0], 8, 8, 1).astype('float32')
    X_test_sub = tf.keras.utils.normalize(X_test_sub, axis=1)
    # cross validation
    print("\n________The cross validation:________")
    cross_val(ax_sub, ay_sub, model_path)
    #  confusion matrix
    print("\n________The confusion matrix________")
    confusion_matrix(model_path, X_train_sub, y_train_sub, X_test_sub, y_test_sub)
    #  ROC
    draw_roc(model_path, X_train_sub, y_train_sub, X_test_sub, y_test_sub)


def test_dnn(ax_sub, ay_sub, X_train_sub, y_train_sub, X_test_sub, y_test_sub):
    print("\n\n********DNN********")
    # model path:
    model_path = 'tf_dnn_model'  # tf is a keyword, please make sure begin with.
    # cross validation
    print("\n________The cross validation:________")
    cross_val(ax_sub, ay_sub, model_path)
    #  confusion matrix
    print("\n________The confusion matrix________")
    confusion_matrix(model_path, X_train_sub, y_train_sub, X_test_sub, y_test_sub)
    #  ROC
    draw_roc(model_path, X_train_sub, y_train_sub, X_test_sub, y_test_sub)


def self_knn():
    print("\n\n********self-write.KNN********")
    print("this section cannot save model, due to self-writing knn")
    print("it will takes about 3 min")
    cross_val(ax, ay, "nil")  # nil is a keyword don't change here.

    # confusion matrix
    print("\ntraining! please wait (approximately 40s)")
    knn_predictions, knn_pro = knnSelf(X_test, 5, X_train, y_train)  # return predict result and probably
    self_cul_matrix(knn_predictions, X_test, y_test)

    # ROC
    self_draw_roc(X_test, y_test, knn_pro)


def call_and_view_neural_network(model_path, ax_sub, ay_sub):
    model = load_myModel(model_path)
    predict_ans = []
    if model_path.startswith('tf_cnn'):
        ax_sub = ax_sub.reshape(ax_sub.shape[0], 8, 8, 1).astype('float32')
        ax_sub = tf.keras.utils.normalize(ax_sub, axis=1)
        for ans in model.predict(ax_sub):
            predict_ans.append(np.argmax(ans))
        print("the predict list for %s" % model_path)
        print(predict_ans)
        test_loss, test_acc = model.evaluate(ax_sub, ay_sub, verbose=0)  # testing
        print('\nTest accuracy:', test_acc)
        print("**************************")
    else:
        for ans in model.predict(ax_sub):
            predict_ans.append(np.argmax(ans))
        print("the predict list for %s" % model_path)
        print(predict_ans)
        test_loss, test_acc = model.evaluate(ax_sub, ay_sub, verbose=0)  # testing
        print('\nTest accuracy:', test_acc)
        print("**************************")


if __name__ == '__main__':
    # load digits and print TensorFlow version
    digits = load_digits()
    ax = digits.data / 16
    ay = digits.target
    # check current version
    print('The TensorFlow version is {}.'.format(tf.__version__))
    print('The scikit-learn version is {}.'.format(sklearn.__version__))
    print('The joblib version is {}.'.format(joblib.__version__))
    print('The np version is {}.'.format(np.__version__))
    # dataset setting
    network_epochs = 10
    X_train, X_test, y_train, y_test = train_test_split(ax, ay, test_size=0.3)

    # call trained model and show them
    call_and_view_neural_network('tf_cnn_model', ax, ay)
    call_and_view_neural_network('tf_dnn_model', ax, ay)

    # call testing methods
    test_dnn(ax, ay, X_train, y_train, X_test, y_test)
    test_cnn(ax, ay, X_train, y_train, X_test, y_test)
    test_knn(ax, ay, X_train, y_train, X_test, y_test)
    self_knn()
    print("Thanks!")
