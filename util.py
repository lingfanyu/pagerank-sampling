import re
from collections import defaultdict as ddict


def read_meta(f):
    # read # nodes, # edges
    f.readline()
    f.readline()
    line = f.readline()
    f.readline()
    m = re.match(r'^# Nodes: (\d*) Edges: (\d*)$', line.strip())
    n_vertex = int(m.group(1))
    n_edge = int(m.group(2))
    #print("# vertex: {}, # edge: {}".format(n_vertex, n_edge))
    return n_vertex, n_edge


def get_num_vertex(filename='web-Stanford.txt'):
    with open(filename, 'r') as f:
        # read # nodes, # edges
        return read_meta(f)[0]


def read_edges(f, n_edge, add_reverse=False):
    if add_reverse:
        edges = ddict(set)
        def add_edge(edges, src, dst):
            edges[src].add(dst)
    else:
        edges = ddict(list)
        def add_edge(edges, src, dst):
            edges[src].append(dst)

    for _ in range(n_edge):
        line = f.readline()
        src, dst = line.split()
        src, dst = int(src) - 1, int(dst) - 1
        add_edge(edges, src, dst)
        # make the graph undirected
        if add_reverse:
            add_edge(edges, dst, src)
    return edges


def read_vertex(f, n_vertex):
    vertices = []
    count = []
    for _ in range(n_vertex):
        line = f.readline()
        v, c = line.split()
        vertices.append(int(v) - 1)
        count.append(int(c))
    return vertices, count


def load_original_graph(filename='web-Stanford.txt'):
    # parse file
    with open(filename, 'r') as f:
        n_vertex, n_edge = read_meta(f)
        edges = read_edges(f, n_edge, add_reverse=True)
        return n_vertex, edges


def read_indices_values(f, n_edge, count):
    indices = []
    values = []
    for _ in range(n_edge):
        line = f.readline()
        src, dst = line.split()
        src, dst = int(src) - 1, int(dst) - 1
        indices.append([src, dst])
        values.append(1.0 / count[dst])
    return indices, values


def read_indices(f, n_edge):
    indices = []
    for _ in range(n_edge):
        line = f.readline()
        src, dst = line.split()
        src, dst = int(src) - 1, int(dst) - 1
        indices.append([src, dst])
    return indices


def load_sampled_graph(filename):
    with open(filename, 'r') as f:
        n_vertex, n_edge = read_meta(f)
        vertices, count = read_vertex(f, n_vertex)
        # sanity check
        assert(f.readline().strip() == '#')
        indices, values = read_indices_values(f, n_edge, count)
        return vertices, indices, values


def convert_to_sparse_M(edges, sort_indices=True):
    indices = []
    values = []
    for src in edges:
        dsts = edges[src]
        v = 1.0 / len(dsts)
        for dst in dsts:
            # note: M_ij is edge j to i divided by num out of j
            indices.append([dst, src])
            values.append(v)
    # reorder indices
    if sort_indices:
        M = zip(indices, values)
        M = sorted(M, key=lambda x:x[0])
        indices, values = zip(*M)

    return indices, values

def serialize(fname, indices):
    with open(fname, 'w') as f:
        for ind in indices:
            f.write("{} {}\n".format(ind[0], ind[1]))

def load_full_graph(filename='web-Stanford.txt', sort_indices=True):
    n_vertex, edges = load_original_graph(filename)
    indices, values = convert_to_sparse_M(edges, sort_indices)
    return n_vertex, indices, values


def elapse(start, end, msg = None):
    if msg is not None:
        print(str(msg) + ":")
    print("{} s".format(end-start))
