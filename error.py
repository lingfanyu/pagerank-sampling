import numpy as np


def error(pr_full, pr_sample):
    pr_full = np.array(pr_full)
    pr_sample = np.array(pr_sample)
    return np.abs(pr_sample - pr_full) / pr_full


if __name__ == '__main__':
    import sys
    f1 = "result_LCC_full.txt"
    assert(len(sys.argv) > 1)
    f2 = sys.argv[1]

    with open(f1, "r") as f:
        pr_full = map(float, list(f))
        print(len(pr_full))

    with open(f2, "r") as f:
        f.readline()
        pr_sample = map(float, list(f))
        print(len(pr_sample))

    # load vertex degree
    with open('vertex_degree.txt') as f:
        degree = map(int, f)

    vertex = np.array(range(1, len(degree) + 1))
    degree = np.array(degree)
    dangling = vertex[degree < 3]

    """
    # nodes in largest connected component
    with open('largest_cc.txt') as f:
        nodes = map(int, f)
        nodes = np.array(nodes)
    """

    # dangling nodes in largest CC
    #nodes = np.intersect1d(dangling, nodes)

    # non-dangling nodes in largest CC
    #nodes = np.setdiff1d(nodes, dangling)

    # convert to 0-index
    #nodes = nodes - 1

    delta = error(pr_full, pr_sample)
    #delta = delta[nodes]

    delta = np.sort(delta)
    n = len(delta)
    print(delta[int(n*0.01)])
    print(delta[int(n*0.1)])
    print(delta[int(n*0.2)])
    print(delta[int(n*0.5)])
    print(delta[int(n*0.8)])
    print(delta[int(n*0.9)])
    print(delta[int(n*0.99)])
