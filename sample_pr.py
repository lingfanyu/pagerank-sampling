import tensorflow as tf
import argparse
import random
from util import read_meta, load_sampled_graph
import numpy as np
import time

def main(args):
    output = "results/{}_{}_{}.txt".format(args.output, args.shuffle and "shuffle" or "no_shuffle", args.percent)
    print(output)
    d = args.damping_factor

    # read # of total vertex
    with open(args.dataset, 'r') as f:
        n_vertex, _ = read_meta(f)

    # global page rank value array
    global_pr = np.array([[1.0 / n_vertex]] * n_vertex, dtype=np.float32)

    # define dataflow graph
    with tf.device('/device:GPU:{}'.format(args.gpu)):
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

    interval = args.samples / 10
    if interval <= 0:
        interval = 1
    sample_order = range(args.samples)

    for epoch in range(args.epochs):
        print("epoch {} / {}".format(epoch, args.epochs))

        pr_buffer = global_pr.copy()

        if args.shuffle:
            # shuffle samples
            random.shuffle(sample_order)

        count = 0
        for sample_idx in sample_order:
            # sample batches and create dataflow graph
            ver, ind, val = load_sampled_graph("samples{}/sample_{}.txt".format(args.percent, sample_idx))

            nver = len(ver)
            # one iteration
            pr_value = sess.run(new_pr,
                    feed_dict={local_pr: pr_buffer[ver],
                               indices: np.array(ind, np.int64),
                               values: val,
                               dense_shape: np.array([nver, nver], np.int64),
                               n_sampled: nver})

            # scatter update local buffer
            pr_buffer[ver] = pr_value

            # write out current pr
            if count % interval == 0:
                with open(output, "w") as f:
                    f.write('epoch {} sample {} {}\n'.format(epoch, count, sample_idx))
                    for v in pr_buffer:
                        f.write(str(v[0]) + '\n')

            print("epoch {} sample {} {}\t{}".format(epoch, count, sample_idx, np.sum(global_pr)))
            # explicitly remove reference
            del ver, ind, val

            count += 1

        # endfor sample

        norm = np.abs(pr_buffer - global_pr) / global_pr
        norm = np.sort(np.reshape(norm, -1))
        norm99 = norm[int(0.99 * n_vertex)]
        norm50 = norm[int(0.5 * n_vertex)]
        global_pr = pr_buffer
        with open("{}_{}".format(output, epoch), "w") as f:
            f.write('epoch {} sample {} {} {} {}\n'.format(epoch, count, sample_idx, norm50, norm99))
            for v in pr_buffer:
                f.write(str(v[0]) + '\n')
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
    parser.add_argument("--gpu", type=int, default=0,
            help="GPU device to use")
    parser.add_argument("--interval", type=int, default=100,
            help="log interval")
    args = parser.parse_args()
    nsamples = {'01': 10240, '10': 5120, '25': 1024, '50': 512, '90': 512}
    if args.samples is None:
        args.samples = nsamples[args.percent]
    print(args)
    random.seed(args.seed)
    main(args)

