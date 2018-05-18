import tensorflow as tf
import argparse
import random
from sampler import load_all_samples
import numpy as np
import time

def main(args):
    d = args.damping_factor

    n_vertex, datasets = load_all_samples(args.dataset, args.samples)

    # global page rank value array
    global_pr = np.array([[1.0 / n_vertex]] * n_vertex, dtype=np.float32)

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

    shuffle = range(args.samples)
    for epoch in range(args.epochs):
        print("epoch {} / {}".format(epoch, args.epochs))
        # shuffle samples
        if args.shuffle:
            random.shuffle(shuffle)

        # reset tensorflow graph
        # tf.reset_default_graph()

        pr_buffer = global_pr.copy()

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
    #    time.sleep(1)
        global_pr = pr_buffer
        print("norm: p50 {}, p99 {}".format(norm50, norm99))
        print(np.sum(global_pr))
        with open(args.output, "w") as f:
            for v in global_pr:
                f.write(str(v[0]) + '\n')
        if norm99 < 1e-5:
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
    parser.add_argument("--samples", type=int, default=1000,
            help="total number of batches")
    parser.add_argument("--output", type=str, default="result_aaa.txt",
            help="output file name")
    parser.add_argument("--epochs", type=int, default=1000,
            help="number of epochs")
    parser.add_argument("--shuffle", action="store_true",
            help="whether or not to shuffle")
    args = parser.parse_args()
    print(args)
    random.seed(args.seed)
    main(args)

