import numpy as np

f1 = "pr_result_full.txt"
f2 = "results/result_shuffle_01.txt"

with open(f1, "r") as f:
    pr_full = map(float, list(f))
    print(len(pr_full))

with open(f2, "r") as f:
    f.readline()
    pr_sample = map(float, list(f))
    print(len(pr_sample))

pr_full = np.array(pr_full)
pr_sample = np.array(pr_sample)

delta = np.abs(pr_sample - pr_full) / pr_full
delta = np.sort(delta)
n = len(delta)
print(delta[int(n*0.5)])
print(delta[int(n*0.9)])
print(delta[int(n*0.99)])
