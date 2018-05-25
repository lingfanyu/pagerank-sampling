import tensorflow as tf
import argparse
import os
import random
from util import elapse
import numpy as np
import time

def dump(output, pr_buffer, epoch, count="", sample_idx="", norm50="", norm99=""):
    with open(output, "w") as f:
        f.write('epoch {} sample {} {} {} {}\n'.format(epoch, count, sample_idx, norm50, norm99))
        for v in pr_buffer:
            f.write(str(v[0]) + '\n')

def main(args):
    interval = args.method == "bfs" and 1 or 10
    output = "{}_{}_{}.txt".format(args.output, args.shuffle and "shuffle" or "no_shuffle", args.percent)
    print(output)
    d = args.damping_factor

    # load full graph or meta information about full graph
    if args.method == "uniform":
        from sampler import uniform_sampling
        from util import load_full_graph
        from scipy.sparse.csr import csr_matrix as csr
        n_vertex, full_ind, full_val = load_full_graph(args.dataset, sort_indices=False)
        full_row, full_col = zip(*full_ind)
        full_M = csr((full_val, (full_row, full_col)), shape=(n_vertex, n_vertex))
    elif args.method == "edge":
        from util import read_meta, read_indices
        from sampler import edge_sampling
        with open(args.dataset) as f:
            n_vertex, n_edge = read_meta(f)
            full_indices = read_indices(f, n_edge)
            full_indices = np.array(full_indices)
    else:
        from util import read_meta, load_sampled_graph
        # read # of total vertex
        with open(args.dataset, 'r') as f:
            n_vertex, _ = read_meta(f)

    percent = int(args.percent) / 100.0

    # global page rank value array
    global_pr = np.array([[1.0 / n_vertex]] * n_vertex, dtype=np.float32)

    # define dataflow graph
    with tf.device('/device:GPU:0'):
        n_sampled = tf.placeholder(tf.int32)
        local_pr = tf.placeholder(tf.float32)
        indices = tf.placeholder(tf.int64)
        values = tf.placeholder(tf.float32)
        dense_shape = tf.placeholder(tf.int64)
        m = tf.SparseTensor(indices=indices,
                            values=values,
                            dense_shape=dense_shape)
        new_pr = d * tf.sparse_tensor_dense_matmul(m, local_pr) \
                + tf.reduce_sum(local_pr) * (1 - d) / tf.to_float(n_sampled)
        sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))

    # print interval
    interval = args.samples / 10
    if interval <= 0:
        interval = 1

    # sample traversal order
    sample_order = range(args.samples)

    for epoch in range(args.epochs):
        print("epoch {} / {}".format(epoch, args.epochs))

        pr_buffer = global_pr.copy()

        # shuffle samples if needed
        if args.shuffle:
            random.shuffle(sample_order)

        count = 0
        for sample_idx in sample_order:
            # load or on-the-fly sample one subgraph
            if args.method == "uniform":
                ver, ind, val = uniform_sampling(full_M, n_vertex, percent, sort=False)
            elif args.method == "edge":
                ver, ind, val = edge_sampling(full_indices, percent)
            else:
                ver, ind, val = load_sampled_graph("samples{}/sample_{}.txt".format(args.percent, sample_idx))

            nver = len(ver)
            # run one iteration
            pr_value = sess.run(new_pr,
                    feed_dict={local_pr: pr_buffer[ver],
                               indices: np.array(ind, np.int64),
                               values: val,
                               dense_shape: np.array([nver, nver], np.int64),
                               n_sampled: nver})

            # scatter update local buffer
            pr_buffer[ver] = pr_value

            # write out current pr
            if args.method == "bfs" and count % interval == 0:
                dump(output, pr_buffer, epoch, count, sample_idx)

            if count % interval == 0:
                print("epoch {} sample {} {}\t{}".format(epoch, count, sample_idx, np.sum(global_pr)))

            # explicitly remove reference to release memory
            del ver, ind, val

            count += 1

        # endfor sample

        norm = np.abs(pr_buffer - global_pr) / global_pr
        norm = np.sort(np.reshape(norm, -1))
        norm99 = norm[int(0.99 * n_vertex)]
        norm50 = norm[int(0.5 * n_vertex)]
        dump("{}_{}".format(output, epoch), pr_buffer, epoch, count, norm50, norm99)
        global_pr = pr_buffer
        print("norm: p50 {}, p99 {}".format(norm50, norm99))
        if norm99 < 1e-4:
            break

    # endfor epoch



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="page rank using spmv")
    parser.add_argument("--damping-factor", metavar='d', type=float, default=0.85,
            help="dampling factor")
    parser.add_argument("--dataset", type=str, default='web-Stanford.txt',
            help="dataset to use")
    parser.add_argument("--seed", type=int, default=20180512,
            help="random seed")
    parser.add_argument("--samples", type=int, default=None,
            help="total number of batches")
    parser.add_argument("--output", type=str, default="result",
            help="output file prefix")
    parser.add_argument("--epochs", type=int, default=1000,
            help="number of epochs")
    parser.add_argument("--shuffle", action="store_true",
            help="whether or not to shuffle")
    parser.add_argument("--percent", type=str, default="10",
            help="sampling percent")
    parser.add_argument("--gpu", type=str, default='0',
            help="GPU device to use")
    parser.add_argument("--method", type=str, default="uniform",
            help="sampling method: [bfs|uniform|edge]")
    args = parser.parse_args()
    nsamples = {'01': 10240, '10': 5120, '25': 1024, '50': 512, '90': 512}
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not args.method in ["bfs", "uniform", "edge"]:
        print("Unknown sampling method: {}".format(args.method))
        exit(-1)
    path = args.output + "_" + args.method
    if not os.path.exists(path):
        os.makedirs(path)
    args.output = os.path.join(path, args.output)
    if args.samples is None:
        args.samples = nsamples[args.percent]
    if args.method == "uniform" or args.method == "edge":
        args.shuffle = False
    print(args)
    random.seed(args.seed)
    main(args)

