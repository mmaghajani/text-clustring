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
                a += ((Ni/N) * math.log2(Ni/N))
        b = 0.000000
        for cat in DATA.keys():
            Niw = WORD_DATA.get(word)[cat]
            if Niw is not 0:
                b += ((Niw/Nw) * math.log2(Niw/Nw))
        c = 0.000000
        for cat in DATA.keys():
            Ni = len(DATA.get(cat))
            Niw = WORD_DATA.get(word)[cat]
            Niwbar = Ni - Niw
            if Niwbar is not 0:
                c += ((Niwbar/Nwbar) * math.log2(Niwbar/Nwbar))
        Pw = Nw / N
        Pwbar = Nwbar / N
        IGs[word] = -a + Pw * b + Pwbar * c
    IGs = sorted(IGs.items(), key=lambda x: x[1], reverse=True)
    with open("info_gain.txt", "w") as f:
        cnt = 0
        while cnt < 100:
            f.write(IGs[cnt][0] + "\t\t\t" + str(IGs[cnt][1]) + "\n")
            cnt += 1
        f.close()


def mutual_info():
    MIs = dict()
    N = 8600
    for word in WORD_DATA.keys():
        Nw = WORD_DATA.get(word)["all"]
        Pw = Nw / N
        for cat in DATA.keys():
            Ni = len(DATA.get(cat))
            Niw = WORD_DATA.get(word)[cat]
            if Niw is not 0:
                MI = math.log2(N*(Niw/N)/(Pw*Ni))
            else:
                MI = -1 * math.inf
            if word not in MIs.keys():
                MIs[word] = {cat: MI}
            else:
                MIs.get(word).update({cat: MI})
    MIs = dict(map(lambda x: (x[0], max(x[1].items(), key=operator.itemgetter(1))), MIs.items()))
    MIs = sorted(MIs.items(), key=lambda x: x[1][1], reverse=False)
    pprint.pprint(MIs)


def X_square():
    return None


read_data()
word_data()
info_gain()
# mutual_info()
# X_square()
