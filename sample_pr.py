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

    n_batches = int(args.samples / args.batch_size)

    for epoch in range(args.epochs):
        print("epoch {} / {}".format(epoch, args.epochs))
        # shuffle samples
        shuffle = range(args.samples)
        random.shuffle(shuffle)
        datasets = [datasets[i] for i in shuffle]

        # page rank in mini-batch fashion
        for bz in range(n_batches):
            # reset tensorflow graph
            tf.reset_default_graph()

            print("mini-batch {} / {}".format(bz, n_batches))
            mappings = []
            local_pr = []
            fetch = []

            # sample batches and create dataflow graph
            batch_base = bz * args.batch_size

            for batch_idx in range(args.batch_size):
                v_sampled, indices, values, new2old = datasets[batch_base + batch_idx]
                n_sampled = len(v_sampled)
                mappings.append(new2old)

                # Multiply M with damping factor
                values = [d * v for v in values]

                # put on CPU first because sparse_reorder does not have a GPU impl...
                with tf.device('/device:CPU:0'):
                    m = tf.SparseTensor(indices=indices,
                                        values=values,
                                        dense_shape=[n_sampled, n_sampled])
                    # sort sparse indices in lexicographical order
                    m = tf.sparse_reorder(m)

                with tf.device('/device:GPU:0'):
                    pr = tf.get_variable("pagerank_{}".format(batch_idx), trainable=False,
                            initializer=tf.constant(global_pr[v_sampled]))
                    local_pr.append(pr)
                    new_pr = tf.sparse_tensor_dense_matmul(m, pr) + (1 - d) / n_sampled
                    delta = tf.abs(new_pr - pr) / pr
                    # make sure assign happens later
                    with tf.control_dependencies([delta]):
                        assignment = pr.assign(new_pr)
                    fetch.append(delta)
                    fetch.append(assignment)

            # iterate until subgraph convergence or 100 iterations
            sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
            # run initializer of all pr variables
            sess.run([pr.initializer for pr in local_pr])
            it = 0
            max_iter = 100
            while len(fetch) > 0 and it < max_iter:
                print("iteration {}:".format(it))
                res = sess.run(fetch)
                next_fetch = []
                for i in range(0, len(fetch), 2):
                    norm = np.sort(np.reshape(res[i], -1))
                    norm = norm[int(len(norm) * 0.99)]
                    print("delta: {}".format(norm))
                    # terminate when relative delta of norm is less than epsilon
                    # convergence: 99 percentile < 1e-4
                    if norm > 1e-5:
                        next_fetch.append(fetch[i])
                        next_fetch.append(fetch[i + 1])
                fetch = next_fetch
                it += 1

            # read out pr for the batch
            pr_value = sess.run(local_pr)

            # simulate a finish-update order
            order = range(args.batch_size)
            random.shuffle(order)
            # update global pr
            for idx in order:
                new2old = mappings[idx]
                for k, v in enumerate(pr_value[idx]):
                    global_pr[new2old[k]] = v

        # end for bz
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
    parser.add_argument("--batch-size", type=int, default=32,
            help="batch size")
    parser.add_argument("--seed", type=int, default=20180512,
            help="random seed")
    parser.add_argument("--samples", type=int, default=1024,
            help="total number of batches")
    parser.add_argument("--output", type=str, default="result.txt",
            help="output file name")
    parser.add_argument("--epochs", type=int, default=10,
            help="number of epochs")
    args = parser.parse_args()
    print(args)
    random.seed(args.seed)
    main(args)

