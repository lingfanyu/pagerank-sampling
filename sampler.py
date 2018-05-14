from util import load_dataset, convert_to_sparse_M
import random
from collections import defaultdict as ddict


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
def filter_and_build_mapping(vertices, edges):
    old = sorted(vertices)
    n = len(old)
    new = range(n)
    # build a mapping from sampled nodes to original nodes
    new2old = {}
    old2new = {}
    for i, j in zip(old, new):
        new2old[j] = i
        old2new[i] = j
    new_edges = ddict(set)
    for src in edges:
        if src in old:
            for dst in edges[src]:
                if dst in old:
                    new_edges[old2new[src]].add(old2new[dst])
    return new_edges, new2old


def load_all_samples(fname, N=64, shuffle=True):
    from util import get_num_vertex, load_dataset, convert_to_sparse_M
    n_vertex = get_num_vertex(fname)
    datasets = []
    for i in range(N):
        print(i)
        n_v, edges = load_dataset("samples/sample_{}.txt".format(i), False)
        v_sampled = set()
        for v in edges.values():
            v_sampled = v_sampled.union(v)
        assert(len(v_sampled) == n_v)
        v_sampled = list(v_sampled)
        e_sampled, new2old = filter_and_build_mapping(v_sampled, edges)
        # create sparse M
        indices, values = convert_to_sparse_M(e_sampled)
        datasets.append((v_sampled, indices, values, new2old))
    if shuffle:
        order = range(N)
        random.shuffle(order)
        datasets = [datasets[i] for i in order]
    return n_vertex, datasets
