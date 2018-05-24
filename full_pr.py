import tensorflow as tf
from util import load_full_graph, elapse
import time
import numpy as np

d = 0.85
n, indices, values = load_full_graph('web-Stanford.txt') #'small_graph.txt')
print("reading dataset done")
with tf.device('/device:GPU:0'):
    m = tf.SparseTensor(indices=indices,
                        values=values,
                        dense_shape=[n, n])
    p = tf.get_variable("pagerank", trainable=False,
            initializer=tf.constant(1.0 / n, shape=[n, 1]))
    new_p = d * tf.sparse_tensor_dense_matmul(m, p) + (1 - d) / n
    delta = tf.abs(new_p - p) / p
    # make sure assign happens later
    with tf.control_dependencies([delta]):
        assignment = p.assign(new_p)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
start = time.time()
sess.run(p.initializer)
i = 0
max_iter = 1000
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
print(np.sum(res[1]))
res = sess.run(p)
with open("result_full.txt", "w") as f:
    for pp in res:
        f.write(str(pp[0]) + '\n')
elapse(start, end, "iteration")

