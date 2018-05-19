import tensorflow as tf
import argparse
import random
from sampler import load_all_samples
import numpy as np
import time



def get_sample_list(args, keep_order=False):
    capacity = {'01': 4096, '10': 5120, '50': 2048, '90': 40}
    if capacity[args.percent] <= args.samples:
        if keep_order:
            return range(capacity[args.percent])
        return random.sample(range(args.samples), capacity[args.percent])
    else:
        return range(args.samples)


def main(args):
    if args.reload:
        shuf = "reload"
    else:
        if args.shuffle:
            shuf = "shuffle"
        else:
            shuf = "no_shuffle"
    output = "{}_{}_{}.txt".format(args.output, shuf, args.percent)
    print(output)
    d = args.damping_factor

    path = "/scratch0/nn-data/lingfan/pagerank/bfs/"
    sample_list = get_sample_list(args, keep_order = True)
    n_vertex, datasets = load_all_samples(args.dataset, sample_list, percent=args.percent, interval=args.interval)

    # global page rank value array
    global_pr = np.array([[1.0 / n_vertex]] * n_vertex, dtype=np.float32)

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

    shuffle = range(len(datasets))
    for epoch in range(args.epochs):
        print("epoch {} / {}".format(epoch, args.epochs))

        if datasets is None:
            # reload dataset
            random_samples = get_sample_list(args)
            _, datasets = load_all_samples(args.dataset, random_samples, percent=args.percent, interval=args.interval)

        pr_buffer = global_pr.copy()

        if args.shuffle:
            # shuffle samples
            random.shuffle(shuffle)

        for sample_idx in shuffle:
            # sample batches and create dataflow graph
            sample = datasets[sample_idx]

            # one iteration
            pr_value = sess.run(new_pr,
                    feed_dict={local_pr: pr_buffer[sample[0]],
                               indices: np.array(sample[1], np.int64),
                               values: sample[2],
                               dense_shape: np.array([len(sample[0]), len(sample[0])], np.int64),
                               n_sampled: len(sample[0])})

            # scatter update local buffer
            pr_buffer[sample[0]] = pr_value

        # endfor sample

        norm = np.abs(pr_buffer - global_pr) / global_pr
        norm = np.sort(np.reshape(norm, -1))
        norm99 = norm[int(0.99 * n_vertex)]
        norm50 = norm[int(0.5 * n_vertex)]
        global_pr = pr_buffer
        print("norm: p50 {}, p99 {}".format(norm50, norm99))
        print(np.sum(global_pr))
        with open(output, "w") as f:
            f.write(str(epoch) + '\n')
            f.write("norm: p50 {}, p99 {}\n".format(norm50, norm99))
            for v in global_pr:
                f.write(str(v[0]) + '\n')
        if norm99 < 1e-5:
            break

        if args.reload:
            del datasets
            datasets = None


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
    parser.add_argument("--reload", action="store_true",
            help="whether or not to reload dataset")
    args = parser.parse_args()
    nsamples = {'01': 10240, '10': 5120, '50': 2048, '90': 1024}
    if args.samples is None:
        args.samples = nsamples[args.percent]
    if args.reload:
        args.shuffle = True
    print(args)
    random.seed(args.seed)
    main(args)

