import numpy as np


def error(pr_full, pr_sample):
    pr_full = np.array(pr_full)
    pr_sample = np.array(pr_sample)
    return np.abs(pr_sample - pr_full) / pr_full


if __name__ == '__main__':
    import sys
    f1 = "result_full.txt"
    assert(len(sys.argv) > 1)
    f2 = sys.argv[1]

    with open(f1, "r") as f:
        pr_full = map(float, list(f))
        print(len(pr_full))

    with open(f2, "r") as f:
        f.readline()
        pr_sample = map(float, list(f))
        print(len(pr_sample))

    delta = error(pr_full, pr_sample)

    delta = np.sort(delta)
    n = len(delta)
    print(delta[int(n*0.01)])
    print(delta[int(n*0.1)])
    print(delta[int(n*0.2)])
    print(delta[int(n*0.5)])
    print(delta[int(n*0.9)])
    print(delta[int(n*0.99)])
