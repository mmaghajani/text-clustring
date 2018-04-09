import operator
import pprint
from sklearn import svm
import math
import random
import jupyter
import matplotlib
import sympy

NUMBER_OF_LINE_IN_RAW_DATA = 8600
DATA = dict()  # A dictionary of sets for each class
WORD_DATA = dict()


def read_data():
    global DATA
    file_path = "raw_data.txt"
    cnt = 0
    with open(file_path) as fp:
        while cnt < NUMBER_OF_LINE_IN_RAW_DATA:
            line = fp.readline()
            cnt += 1
            data = line.split("@@@@@@@@@@")
            cat = data[0]
            doc = data[1]
            if cat not in DATA.keys():
                DATA[cat] = set()
                DATA[cat].add(doc)
            else:
                DATA.get(cat).add(doc)


def word_data():
    global WORD_DATA
    domains = list(DATA.keys())
    for domain in domains:
        docs = DATA.get(domain)
        for doc in docs:
            words = set(doc.split(" "))
            for word in words:
                if word not in WORD_DATA.keys():
                    a = domains[0]
                    b = domains[1]
                    c = domains[2]
                    d = domains[3]
                    e = domains[4]
                    WORD_DATA[word] = {"all": 1, a: 0, b: 0, c: 0, d: 0, e: 0}
                    WORD_DATA[word][domain] += 1
                else:
                    WORD_DATA[word][domain] += 1
                    WORD_DATA[word]["all"] += 1
    while True:
        try:
            WORD_DATA.pop("")
        except KeyError as e:
            break
    WORD_DATA.pop("\n")


def info_gain():
    N = 8600
    IGs = dict()
    for word in WORD_DATA.keys():
        Nw = WORD_DATA.get(word)["all"]
        Nwbar = N - Nw
        a = 0.000000
        for cat in DATA.keys():
            Ni = len(DATA.get(cat))
            if Ni is not 0:
                a += ((Ni / N) * math.log2(Ni / N))
        b = 0.000000
        for cat in DATA.keys():
            Niw = WORD_DATA.get(word)[cat]
            if Niw is not 0:
                b += ((Niw / Nw) * math.log2(Niw / Nw))
        c = 0.000000
        for cat in DATA.keys():
            Ni = len(DATA.get(cat))
            Niw = WORD_DATA.get(word)[cat]
            Niwbar = Ni - Niw
            if Niwbar is not 0:
                c += ((Niwbar / Nwbar) * math.log2(Niwbar / Nwbar))
        Pw = Nw / N
        Pwbar = Nwbar / N
        IGs[word] = -a + Pw * b + Pwbar * c
    IGs = sorted(IGs.items(), key=lambda x: x[1], reverse=True)
    with open("info_gain.csv", "w") as f:
        cnt = 0
        f.write("score" + "\t" + "word\n")
        while cnt < 100:
            f.write(str(IGs[cnt][1]) + "\t" + IGs[cnt][0] + "\n")
            cnt += 1
        f.close()
    return list(map(lambda x: x[0], dict(IGs[:100]).items()))


def mutual_info():
    MIs = dict()
    N = 8600
    score = dict()
    for word in WORD_DATA.keys():
        Nw = WORD_DATA.get(word)["all"]
        Nwbar = N - Nw
        for cat in DATA.keys():
            Ni = len(DATA.get(cat))
            Niw = WORD_DATA.get(word)[cat]
            Niwbar = Ni - Niw
            Nibar = N - Ni
            Nibarw = Nw - Niw
            Nibarwbar = Nibar - Nibarw
            a = 0.0000000000
            b = 0.0000000000
            c = 0.0000000000
            d = 0.0000000000
            if Niw is not 0:
                a = (Niw / N) * math.log2((N * Niw) / (Nw * Ni))
            if Niwbar is not 0:
                b = (Niwbar / N) * math.log2((N * Niwbar) / (Nwbar * Ni))
            if Nibarw is not 0:
                c = (Nibarw / N) * math.log2((N * Nibarw) / (Nw * Nibar))
            if Nibarwbar is not 0:
                d = (Nibarwbar / N) * math.log2((N * Nibarwbar) / (Nwbar * Nibar))
            MI = a + b + c + d
            if word not in MIs.keys():
                MIs[word] = {cat: MI}
            else:
                MIs.get(word).update({cat: MI})
        s = 0
        for cat in DATA.keys():
            Ni = len(DATA.get(cat))
            Pci = Ni / N
            s += (MIs.get(word)[cat] * Pci)
        score[word] = s
    MIs = dict(map(lambda x: (x[0], (score[x[0]],
                                     max(x[1].items(), key=operator.itemgetter(1)))), MIs.items()))
    MIs = sorted(MIs.items(), key=lambda x: x[1][0], reverse=True)
    with open("mutual_info.csv", "w") as f:
        cnt = 0
        f.write("max class" + "\t" + "max class score" + "\t" + "score" + "\t" + "word\n")
        while cnt < 100:
            f.write(str(MIs[cnt][1][1][0]) + "\t" + str(MIs[cnt][1][1][1]) +
                    "\t" + str(MIs[cnt][1][0]) + "\t" + MIs[cnt][0] + "\n")
            cnt += 1
        f.close()
    return list(map(lambda x: x[0], dict(MIs[:100]).items()))


def X_square():
    Xs = dict()
    N = 8600
    score = dict()
    for word in WORD_DATA.keys():
        Nw = WORD_DATA.get(word)["all"]
        for cat in DATA.keys():
            Ni = len(DATA.get(cat))
            Niw = WORD_DATA.get(word)[cat]
            Niwbar = Ni - Niw
            Nibar = N - Ni
            Nibarw = Nw - Niw
            Nibarwbar = Nibar - Nibarw

            a = N * ((Niw * Nibarwbar - Niwbar * Nibarw) ** 2)
            b = (Niw + Niwbar) * (Nibarw + Nibarwbar) * (Niw + Nibarw) * (Niwbar + Nibarwbar)
            X = a / b
            if word not in Xs.keys():
                Xs[word] = {cat: X}
            else:
                Xs.get(word).update({cat: X})
        s = 0
        for cat in DATA.keys():
            Ni = len(DATA.get(cat))
            Pci = Ni / N
            s += (Xs.get(word)[cat] * Pci)
        score[word] = s
    Xs = dict(map(lambda x: (x[0], (score[x[0]],
                                    max(x[1].items(), key=operator.itemgetter(1)))), Xs.items()))
    Xs = sorted(Xs.items(), key=lambda x: x[1][0], reverse=True)
    with open("chi_square.csv", "w") as f:
        cnt = 0
        f.write("max class" + "\t" + "max class score" + "\t" + "score" + "\t" + "word\n")
        while cnt < 100:
            f.write(str(Xs[cnt][1][1][0]) + "\t" + str(Xs[cnt][1][1][1]) +
                    "\t" + str(Xs[cnt][1][0]) + "\t" + Xs[cnt][0] + "\n")
            cnt += 1
        f.close()
    return list(map(lambda x: x[0], dict(Xs[:100]).items()))


def get_frequency_words():
    new_word_data = sorted(WORD_DATA.items(), key=lambda x: x[1]["all"], reverse=True)
    return list(map(lambda x: x[0], dict(new_word_data[:1000]).items()))


def classify(train, test):
    train_label = list(map(lambda x: x[-1:][0], train))
    final_train = list(map(lambda x: x[:-1], train))
    test_label = list(map(lambda x: x[-1:][0], test))
    final_test = list(map(lambda x: x[:-1], test))
    clf = svm.LinearSVC(penalty='l1', loss='squared_hinge', dual=False, random_state=0)
    clf.fit(final_train, train_label)
    predicted_label = clf.predict(final_test)
    cnt = 0

    for i in range(0, len(predicted_label)-1):
        if predicted_label[i] == test_label[i]:
            cnt += 1

    return cnt


def cross_validation(k, features):
    vectored_data = list()
    for cat in DATA.keys():
        for doc in DATA.get(cat):
            vec = dict((el, 0) for el in features)
            count = 0
            for word in doc.split(" "):
                if word in vec.keys():
                    vec[word] += 1
                    count += 1
            if count == 0:
                count = 1
            vec = dict(map(lambda x: (x[0], x[1] / count), vec.items()))
            vec["label"] = cat
            vectored_data.append(list(vec.values()))
    vectored_data = sorted(vectored_data, key=lambda k: random.random())

    cnt = 0

    x = NUMBER_OF_LINE_IN_RAW_DATA/k
    for i in range(0, k):
        train = list(vectored_data[:int(x*i)] +
                     vectored_data[int(x*(i+1)):])
        test = list(vectored_data[int(x*i): int(x*(i+1))])
        cnt += classify(train, test)

    return cnt / NUMBER_OF_LINE_IN_RAW_DATA


def read_features(file_path):
    cnt = 0
    features = list()
    with open(file_path) as fp:
        line = "s"
        while line:
            line = fp.readline()
            cnt += 1
            features.append(line.strip())
    return features


read_data()
word_data()
# features0 = get_frequency_words()
features0 = read_features("1000.txt")
# features0.append("label")
# print(features0)
features1 = read_features("infogain.txt")
# features1 = info_gain()
# features1.append("label")
# print(features1)
# features2 = mutual_info()
features2 = read_features("mutual.txt")
# features2.append("label")
# print(features2)
# features3 = X_square()
features3 = read_features("chi.txt")
# features3.append("label")
# print(features3)


# ------------------------------- PART II --------------------------------- #
print(cross_validation(5, features0))
print(cross_validation(5, features1))
print(cross_validation(5, features2))
print(cross_validation(5, features3))
