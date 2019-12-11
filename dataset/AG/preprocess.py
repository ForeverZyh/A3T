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
        ave = 0
        precent = 0
        for row in csv_reader:
            y.append(int(row[0]) - 1)
            s = row[1] + row[2]
            s = s.lower()
            ave += len(s)
            cnt = 0
            x = [0] * 300
            if (len(s) - 1) <= 200:
                precent += 1
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

        print(ave * 1.0 / len(y), precent * 1.0 / len(y))

        return np.array(X), np.array(y)


X, y = openfile("./test.csv")
print(X[:10, :10])
np.save("X_test", X)
np.save("y_test", y)
X, y = openfile("./train.csv")
np.save("X_train", X)
np.save("y_train", y)
print(dict)
