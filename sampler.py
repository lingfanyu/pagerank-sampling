import random
from collections import defaultdict as ddict
import time
from util import elapse, convert_to_sparse_M
import numpy as np

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


# uniform sampling by slicing scipy sparse matrix
def uniform_sampling(M, n_vertex, percent, sort=False):
    n_sampled = int(n_vertex * percent)
    v_sampled = np.array(sorted(random.sample(range(n_vertex), n_sampled)))
    S = M
    S = S[v_sampled, :]
    S = S[:, v_sampled]

    # filter out vertices with no edges
    ne = np.diff(S.indptr) != 0
    S = S[ne, :]
    S = S[:, ne]
    v_sampled = v_sampled[ne]
    S = S.tocoo()
    indices = zip(S.row, S.col)

    """
    # build new sparse M
    edges = ddict(list)
    for src, dst in indices:
        edges[src].append(dst)
    indices, values = convert_to_sparse_M(edges, sort)
    """

    # count outging edges
    _, inv, count = np.unique(S.col, return_inverse=True, return_counts=True)
    values = 1.0 / count[inv]

    return v_sampled, indices, values

def edge_sampling(indices_full, percent):
    n_edge = len(indices_full)

    # edge uniform sampling
    samples = random.sample(range(n_edge), int(percent * n_edge))
    sampled_edges = indices_full[samples]

    # get sampled vertices
    v_sampled, e_sampled = np.unique(sampled_edges, return_inverse=True)
    n_sampled = len(v_sampled)

    # get new edges (contiguous range)
    e_sampled = np.reshape(e_sampled, (-1, 2))

    # add reverse edges
    e_sampled = np.concatenate((e_sampled, e_sampled[:, [1,0]]))

    # deduplicate and sort
    indices = np.unique(e_sampled, axis=0)

    # count outgoing edges
    _, inv, count = np.unique(indices[:, 1], return_inverse=True, return_counts=True)

    # calculate sparse M values
    values = 1.0 / count[inv]
    return v_sampled, indices, values


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
