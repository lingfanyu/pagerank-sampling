from degree_cdf import count_degree
import sys
from error import error
import numpy as np

def degree_error():
    assert(len(sys.argv) > 1)
    fname = sys.argv[1]

    _, degree = count_degree()

    with open(fname) as f:
        f.readline()
        pr = map(float, list(f))

    with open('result_full.txt') as f:
        pr_full = map(float, list(f))

    distance = error(pr_full, pr)

    err = np.bincount(degree, weights=distance)
    degree = np.unique(degree)
    err = err[degree]
    err = err / np.sum(distance)
    for i in zip(degree, err):
        print(i)

    print(np.sum(err))

def degree_sample():
    from collections import defaultdict as ddict
    assert(len(sys.argv) > 1)
    fname = 'vertex_degree.txt'

    with open(fname) as f:
        # add -1 for future 1-indexed loop up
        degree = [-1] + map(lambda x: int(x.strip()), f)

    fname = sys.argv[1]

    d = ddict(list)
    with open(fname) as f:
        for line in f:
            if line[0] == '#':
                break
            line = map(int, line.split())
            d[degree[line[0]]].append(line[1])

    keys = sorted(d.keys())
    for k in keys:
        c = sorted(d[k])
        n = len(c)
        print("{}:\t {} {} {} {} {} {} {}".format(k, c[int(n*0.01)], c[int(n*0.1)], c[int(n*0.25)], c[int(n*0.5)], c[int(n*0.75)], c[int(n*0.9)], c[int(n*0.99)]))


if __name__ == '__main__':
    degree_sample()

