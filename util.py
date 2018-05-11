import re
import pickle
import os
from collections import defaultdict as ddict

def load_dataset(bfile, filename='web-Stanford.txt'):
    # if parsed file exists
    if os.path.isfile(bfile):
        with open(bfile, 'rb') as f:
            data = pickle.load(f)
            return data

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

        edges = ddict(set)
        for _ in range(n_edge):
            line = f.readline()
            src, dst = line.split()
            src, dst = int(src) - 1, int(dst) - 1
            edges[src].add(dst)

    indices = []
    values = []
    for src in edges:
        dsts = edges[src]
        v = 1.0 / len(dsts)
        for dst in dsts:
            # note: M_ij is edge j to i divided by num out of j
            indices.append([dst, src])
            values.append(v)

    data = (n_vertex, indices, values)

    # store parsed result
    with open(bfile, 'wb') as f:
        pickle.dump(data, f)

    return data
