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


    for epoch in range(args.epochs):
        print("epoch {} / {}".format(epoch, args.epochs))
        # shuffle samples
        shuffle = range(args.samples)
        random.shuffle(shuffle)
        datasets = [datasets[i] for i in shuffle]

        # pr buffer
        pr_buffer = global_pr.copy()

        # page rank in mini-batch fashion
        for sample_idx in range(args.samples):
            # reset tensorflow graph
            tf.reset_default_graph()

            print("sample {} / {}".format(sample_idx, args.samples))

            # sample batches and create dataflow graph

            v_sampled, indices, values, new2old = datasets[sample_idx]
            n_sampled = len(v_sampled)

            # Multiply M with damping factor
            values = [d * v for v in values]

            # put on CPU first because sparse_reorder does not have a GPU impl...
            with tf.device('/device:CPU:0'):
                m = tf.SparseTensor(indices=indices,
                                    values=values,
                                    dense_shape=[n_sampled, n_sampled])
                # sort sparse indices in lexicographical order
                m = tf.sparse_reorder(m)

            fetch = []
            with tf.device('/device:GPU:0'):
                pr = tf.constant(global_pr[v_sampled])
                new_pr = tf.sparse_tensor_dense_matmul(m, pr) + (1 - d) / n_sampled
                # make sure assign happens later

            # iterate until subgraph convergence or 100 iterations
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
            pr_value = sess.run(new_pr)

            # write out
            for k, v in enumerate(pr_value):
                pr_buffer[new2old[k]] = v

        # end for sample
        norm = np.abs(pr_buffer - global_pr) / global_pr
        norm = np.sort(np.reshape(norm, -1))
        print(len(norm))
        print(int(0.99 * n_vertex))
        norm99 = norm[int(0.99 * n_vertex)]
        global_pr = pr_buffer
        print("norm: {}".format(norm99))
        time.sleep(1)
        if norm99 < 1e-5:
            break

    # end for epoch

    print(global_pr)
    print(np.sum(global_pr))
    with open(args.output, "w") as f:
        for v in global_pr:
            f.write(str(v[0]) + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="page rank using spmv")
    parser.add_argument("--damping-factor", metavar='d', type=float, default=0.85,
            help="dampling factor")
    parser.add_argument("--dataset", type=str, default='web-Stanford.txt',
            help="dataset to use")
    parser.add_argument("--seed", type=int, default=20180512,
            help="random seed")
    parser.add_argument("--samples", type=int, default=8192,
            help="total number of batches")
    parser.add_argument("--output", type=str, default="result.txt",
            help="output file name")
    parser.add_argument("--epochs", type=int, default=1000,
            help="number of epochs")
    args = parser.parse_args()
    print(args)
    random.seed(args.seed)
    main(args)

