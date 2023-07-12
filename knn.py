from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import pickle
from joblib import dump, load

# please load the code by UTF-8


digits = load_digits()
print("load successfully, the number of sample & features:", digits.data.shape)
ax = digits.data
ay = digits.target

# f1 show basic information
print("\n-------f1 show basic information-------")
print("the number of instances:", len(ax))
print("the number of features of each instance:", len(digits.data[0]))
feature = digits.images.reshape(-1, 64)

i = 0
j = 0
maxNum = 0
minNum = 20
while i < len(digits.data[0]):
    while j < len(ax):
        if feature[j][i] > maxNum:
            maxNum = feature[j][i]
        if feature[j][i] < minNum:
            minNum = feature[j][i]
        j += 1
    print("the pixel in ({},{}) range:(integers) {:.0f} to {:.0f}".format(i//8+1,i%8+1, minNum, maxNum))
    i += 1
    j = 0
    minNum = 200
    maxNum = 0

classList = []
for i in ay:
    if not i in classList:
        classList.append(i)
print("\nThe number of classes: ", len(classList))

print("detail:")
count = 0
for seq in classList:
    for i in ay:
        if i == seq:
            count += 1
    print("the number of data entries for {}: {:.0f}".format(seq, count))
    count = 0

# split train sets and test sets
sizeOfTest = 0.4
X_train, X_test, y_train, y_test = train_test_split(ax, ay, test_size=sizeOfTest)
print("\nX_train & y_train are train set,\ny_train & y_test are test size. \nTest size is {:.2%}".format(sizeOfTest))

# f2 calling an algorithm from the machine learning libraries
print("\n-------f2: calling an knn algorithm-------")
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
res_knn = knn.predict(X_test)

# save modle
s = pickle.dumps(knn)
dump(knn, 'knn.joblib')

i = 0
corr = 0
while i < len(res_knn):
    if y_test[i] == res_knn[i]:
        corr += 1
    i += 1
print("Correct testing sample:", corr)
print("Total  testing  sample:", len(res_knn))
autoKnnPercetage = corr/len(res_knn)
print("Percentage of correct by useing auto-knn in test set {:.2%}".format(autoKnnPercetage))

# 3 machine learning algorithm by myself
print("\n-------f3: myself knn algorithm-------")

# function of finding Euclidean distance
def distance(d1, d2):
    dis_res = 0

    for feature_index in range(0,64):
        dis_res += ((d1[feature_index]) - (d2[feature_index])) ** 2

    return dis_res ** 0.5

# 1 find distance
def knnSelf(data, k):
    num = 0
    res = [
        {"result": y_train[num], "distance": distance(data, X_train[num])}
        for num in range(0, len(X_train))
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
    temp = 0
    for i in range(0, 10):
        if result[i] > max_avg:
            max_avg = result[i]
            temp = i
    # print(max_avg,temp)
    # print(result)
    # print("____result_____\n")
    return temp


# please input "k" for k nearest neighbor here,
# 5 is the best k in my test case.
k = 5
corr = 0
for sample in range(0, len(y_test)):
    if knnSelf(X_test[sample], k) == y_test[sample]:
        corr += 1
print("Correct testing sample:", corr)
print("Total  testing  sample:", len(X_test))
myKnnPercetage = corr/len(res_knn)
print("Percentage of correct by knn wrote by myself in test set {:.2%}".format(myKnnPercetage))

# f4 train and test errors

print("\n-------f4: train and test errors-------")
print("test error:")
print("using auto-knn in test set {:.2%}".format(autoKnnPercetage))

train_res = knn.predict(X_train)
corr = 0
for i in range(0, len(X_train)):
    if train_res[i] == y_train[i]:
        corr += 1
print("train error:")
print("using auto-knn in train set {:.2%}\n".format(corr/len(train_res)))


print("test error:")
print("using my-knn in test set {:.2%}".format(myKnnPercetage))

corr = 0
for sample in range(0, len(y_train)):
    if knnSelf(X_train[sample], k) == y_train[sample]:
        corr += 1
print("train error:")
print("using my-knn in train set {:.2%}".format(corr/len(y_train)))


print("\n-------f5: train and test errors-------")
total = len(X_test)
while 1:
    # get index form key board
    index_num = input("The dataset has {} instance(from 0 to {}), \nplease"
                      " input index to check detail".format(total, total-1))
    index_num = int(index_num)
    # get k from key board
    k_user = input("please input k (2 to 63)")
    k_user = int(k_user)

    print("my knn algorithm predict is: {}, result from y is "
          "{}".format(knnSelf(X_test[index_num],k_user),y_test[index_num]))
    print("sklearn knn predict: {} \n".format(res_knn[index_num]))
