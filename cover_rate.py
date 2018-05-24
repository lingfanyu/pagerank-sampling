from util import read_meta, read_vertex
import sys

dataset = 'web-Stanford.txt'

with open(dataset, 'r') as f:
    n_vertex, _ = read_meta(f)

path = sys.argv[1]
n = int(sys.argv[2])

interval = n / 100

vertex = set()

for i in range(n):
    if i % interval == 0:
        print("{}:\t {} / {}".format(i, len(vertex), n_vertex))
    fname = path + "/sample_{}.txt".format(i)
    with open(fname, 'r') as f:
        n_sampled, _ = read_meta(f)
        v_sampled, _ = read_vertex(f, n_sampled)
    vertex = vertex.union(set(v_sampled))

print("cover rate: {} / {}".format(len(vertex), n_vertex))
