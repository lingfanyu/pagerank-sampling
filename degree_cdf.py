from util import read_meta, read_indices
import numpy as np

def count_degree():
    fname = 'web-Stanford.txt'
    with open(fname, 'r') as f:
        n_vertex, n_edge = read_meta(f)
        indices = read_indices(f, n_edge)

    indices = np.array(indices)
    indices = np.concatenate((indices, indices[:, (1, 0)]))
    edges = np.unique(indices, axis=0)
    dsts = edges[:, 1]

    _, degree = np.unique(dsts, return_counts=True)

    unique_degree, count = np.unique(degree, return_counts=True)
    ccount = np.cumsum(count)
    ccount = ccount / float(ccount[-1])

    res = zip(unique_degree, count, ccount)
    return res, degree

if __name__ == '__main__':
    cdf, degree = count_degree()
    for i in cdf:
        print(i)
    with open('vertex_degree.txt', 'w') as f:
        for i in degree:
            f.write("%d\n" % i)

