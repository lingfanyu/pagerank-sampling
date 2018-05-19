import random
from collections import defaultdict as ddict
import time
from util import elapse


def uniform_sampler(n, p, edges):
    vertices = random.sample(range(n), int(n * p))
    return vertices


def bfs_sampler(n, p, edges):
    start = random.randint(0, n - 1)
    n_sample = int(n * p)
    sampled = [start]
    i = 0
    count = 1
    while i < count and count < n_sample:
        s = sampled[i]
        for t in edges[s]:
            if not t in sampled:
                sampled.append(t)
                count += 1
        i += 1
    if count > n_sample:
        sampled = sampled[:n_sample]
    return sampled


# filter edges and build mapping from new vertices to old
def filter_and_build_mapping(old, edges):
    n = len(old)
    new = range(n)
    # build a mapping from sampled nodes to original nodes
    old2new = {}
    for i, j in zip(old, new):
        old2new[i] = j
    new_edges = ddict(set)
    for src in edges:
        if src in old:
            for dst in edges[src]:
                if dst in old:
                    new_edges[old2new[src]].add(old2new[dst])
    return new_edges


def load_all_samples(fname, sample_list, percent="", path="", interval=100):
    import os
    import psutil
    process = psutil.Process(os.getpid())
    unit = 2.0**30
    print(process.memory_info().rss / unit)
    from util import get_num_vertex, load_sampled_graph
    n_vertex = get_num_vertex(fname)
    datasets = []
    for i, file_idx in enumerate(sample_list):
        if i % interval == 0:
            print(i, file_idx)
            print(process.memory_info().rss / unit)
        v_sampled, indices, values = load_sampled_graph("{}samples{}/sample_{}.txt".format(path, percent, file_idx))
        # assert(len(v_sampled) == n_v)

        # create sparse M
        # indices, values = convert_to_sparse_M(e_sampled, sort_indices=False)
        datasets.append((v_sampled, indices, values))
    return n_vertex, datasets
