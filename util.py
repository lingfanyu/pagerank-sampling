import re
from collections import defaultdict as ddict

def get_num_vertex(filename='web-Stanford.txt'):
    with open(filename, 'r') as f:
        # read # nodes, # edges
        f.readline()
        f.readline()
        line = f.readline()
        f.readline()
        m = re.match(r'^# Nodes: (\d*) Edges: \d*$', line.strip())
        n_vertex = int(m.group(1))
        print("# vertex: {}".format(n_vertex))
        return n_vertex


def load_dataset(filename='web-Stanford.txt', add_reverse=True):
    # parse file
    with open(filename, 'r') as f:
        # read # nodes, # edges
        f.readline()
        f.readline()
        line = f.readline()
        f.readline()
        m = re.match(r'^# Nodes: (\d*) Edges: (\d*)$', line.strip())
        n_vertex = int(m.group(1))
        n_edge = int(m.group(2))
        print("# vertex: {}, # edge: {}".format(n_vertex, n_edge))

        edges = ddict(set)
        for _ in range(n_edge):
            line = f.readline()
            src, dst = line.split()
            src, dst = int(src) - 1, int(dst) - 1
            edges[src].add(dst)
            # make the graph undirected
            if add_reverse:
                edges[dst].add(src)

    #sink = set(range(n_vertex)) - set(edges.keys())
    #print("# sink = {}".format(len(sink)))

    return n_vertex, edges


def convert_to_sparse_M(edges):
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
    M = zip(indices, values)
    M = sorted(M, key=lambda x:x[0])
    indices, values = zip(*M)

    return indices, values


def load_full_graph(filename='web-Stanford.txt'):
    n_vertex, edges = load_dataset(filename)
    indices, values = convert_to_sparse_M(edges)

    """
    # add outgoing edges for sink nodes
    for src in sink:
        indices.extend([[dst, src] for dst in range(n_vertex) if dst != src])
    values.extend([1.0 / (n_vertex - 1)] * ((n_vertex - 1) * len(sink)))
    """

    return n_vertex, indices, values


def elapse(start, end, msg = None):
    if msg is not None:
        print(str(msg) + ":")
    print("{} s".format(end-start))
