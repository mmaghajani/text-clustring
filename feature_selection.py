import operator
import sys
import pprint

import math

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
                if word not in WORD_DATA:
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
    with open("info_gain.txt", "w") as f:
        cnt = 0
        f.write("score" + "\t\t\t\t\t\t" + "word\n")
        while cnt < 100:
            f.write(str(IGs[cnt][1]) + "\t\t\t" + IGs[cnt][0] + "\n")
            cnt += 1
        f.close()


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
    with open("mutual_info.txt", "w") as f:
        cnt = 0
        f.write("max class" + "\t\t" + "max class score" + "\t\t\t\t" + "score" + "\t\t\t\t\t\t" + "word\n")
        while cnt < 100:
            f.write(str(MIs[cnt][1][1][0]) + "\t\t\t" + str(MIs[cnt][1][1][1]) +
                    "\t\t\t" + str(MIs[cnt][1][0]) + "\t\t\t" + MIs[cnt][0] + "\n")
            cnt += 1
        f.close()


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
    with open("chi_square.txt", "w") as f:
        cnt = 0
        f.write("max class" + "\t\t" + "max class score" + "\t\t\t\t" + "score" + "\t\t\t\t\t\t" + "word\n")
        while cnt < 100:
            f.write(str(Xs[cnt][1][1][0]) + "\t\t\t" + str(Xs[cnt][1][1][1]) +
                    "\t\t\t" + str(Xs[cnt][1][0]) + "\t\t\t" + Xs[cnt][0] + "\n")
            cnt += 1
        f.close()


read_data()
word_data()
info_gain()
mutual_info()
X_square()
