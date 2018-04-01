import sys
import pprint

NUMBER_OF_LINE_IN_RAW_DATA = 8600
DATA = dict()   # A dictionary of sets for each class


def read_data():
    file_path = "raw_data.txt"
    cnt = 0
    with open(file_path) as fp:
        while cnt < NUMBER_OF_LINE_IN_RAW_DATA:
            line = fp.readline()
            cnt += 1
            data = line.split("@@@@@@@@@@")
            print(cnt)
            cat = data[0]
            doc = data[1]
            if cat not in DATA.keys():
                DATA[cat] = set()
                DATA[cat].add(doc)
            else:
                # print(type(DATA.get(cat)))
                DATA.get(cat).add(doc)

            # pprint.pprint(DATA.keys())
            # cnt += 1
            # if cnt >= 10000:
            #     sys.exit()


def info_gain(data):
    N = 0
    Ni = 0
    Nw = 0
    N_not_w = 0
    Niw = 0
    N_not_iw = 0
    return None


def mutual_info(data):
    return None


def X_square(data):
    return None


read_data()
