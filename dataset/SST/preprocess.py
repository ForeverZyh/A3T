import numpy as np
import csv

dict = {"space": 0}


def map(c):
    if c not in dict:
        dict[c] = len(dict)

    return dict[c]


def openfile(filename):
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        X = []
        y = []
        for row in csv_reader:
            y.append(int(row[0]) - 1)
            s = row[1] + row[2]
            s = s.lower()
            cnt = 0
            x = [0] * 300
            for c in s:
                if cnt == 300:
                    break
                elif not c.isspace():
                    x[cnt] = map(c)
                    cnt += 1
                else:
                    x[cnt] = 0
                    cnt += 1
            X.append(x)

        return np.array(X), np.array(y)


lines = open("./datasetSplit.txt").readlines()
split_map = {}
for line in lines[1:]:
    tmp = line.strip().split(',')
    split_map[int(tmp[0])] = int(tmp[1])

lines = open("./datasetSentences.txt").readlines()
ave = 0
max_len = 0
precent = 0
for line in lines[1:]:
    ave += len(line) - 1
    max_len = max(max_len, len(line) - 1)
    if len(line) - 1 <= 200: precent += 1

print(ave * 1.0 / (len(lines) - 1), max_len, precent * 1.0 / (len(lines) - 1))

lines = open("./sentiment_labels.txt").readlines()
labels = [0.0] * len(lines)
cnt = 0
for line in lines[1:]:
    tmp = line.strip().split('|')
    labels[int(tmp[0])] = float(tmp[1])
    if float(tmp[1]) != 0.5:
        cnt += 1
print(cnt)

lines = open("./dictionary.txt").readlines()
words = {}
for line in lines[1:]:
    tmp = line.strip().split('|')
    words[tmp[0]] = int(tmp[1])

lines = open("./train.txt").readlines()
number = 0
in_dict = 0
in_dict_not_n = set()
for line in lines:
    word_list = line.strip().split()
    for j in range(len(word_list)):
        for k in range(j, len(word_list)):
            phrase = ' '.join(word_list[j:k + 1])
            if phrase in words:
                in_dict += 1
                if labels[words[phrase]] != 0.5:
                    in_dict_not_n.add(phrase)

print(in_dict, len(in_dict_not_n))
