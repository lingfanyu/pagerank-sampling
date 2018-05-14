import tensorflow as tf
from util import load_full_graph, elapse
import time
import numpy as np

d = 0.85
n, indices, values = load_full_graph()
print("reading dataset done")
values = [d * v for v in values]

# put on CPU first because sparse_reorder does not have a GPU impl...
with tf.device('/device:CPU:0'):
    start = time.time()
    m = tf.SparseTensor(indices=indices,
                        values=values,
                        dense_shape=[n, n])
    end = time.time()
    elapse(start, end, "creating sparse tensor")
    # sort sparse indices in lexicographical order
    m = tf.sparse_reorder(m)

with tf.device('/device:GPU:2'):
    p = tf.get_variable("pagerank", trainable=False,
            initializer=tf.constant(1.0 / n, shape=[n, 1]))
    new_p = tf.sparse_tensor_dense_matmul(m, p) + (1 - d) / n
    delta = tf.abs(new_p - p) / p
    # make sure assign happens later
    with tf.control_dependencies([delta]):
        assignment = p.assign(new_p)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
start = time.time()
sess.run(p.initializer)
i = 0
max_iter = 100
while i < max_iter:
    res = sess.run([delta, assignment])
    norm = np.sort(np.reshape(res[0], -1))
    norm = norm[int(n * 0.99)]
    print("iteration {}:".format(i))
    print("99 percentile delta is {}".format(norm))
    # terminate when relative delta of norm is less than epsilon
    # convergence: 99 percentile < 1e-5
    if norm < 1e-5:
        break
    i += 1

end = time.time()
pr = sess.run(p)
print(np.sum(pr))
with open("result_full.txt", "w") as f:
    for p in pr:
        f.write(str(p[0]) + '\n')
elapse(start, end, "iteration")

