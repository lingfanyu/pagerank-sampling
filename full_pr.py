import tensorflow as tf
from util import load_dataset, elapse
import time

d = 0.85
n, indices, values = load_dataset('web.pkl')
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
    m = tf.sparse_reorder(m)

with tf.device('/device:GPU:0'):
    p = tf.constant(1.0 / n, shape=[n, 1])
    delta = tf.constant(1.0)
    step = tf.constant(0)
    # terminate when relative delta of norm is less than epsilon
    cond = lambda step, p, delta : delta > 1e-5
    sum_p = tf.reduce_sum(p)
    def body(step, p, delta):
        last_p = p
        p = tf.sparse_tensor_dense_matmul(m, p) + (1 - d) / n
        # set shape for while_loop shape inference
        p.set_shape([n, 1])
        delta = tf.norm(p - last_p) / tf.norm(last_p)
        return [step + 1, p, delta]

    res = tf.while_loop(cond, body, loop_vars=[step, p, delta], back_prop=False)
    step, pr, delta = res
    sum_pr = tf.reduce_sum(pr)

print("finish dataflow graph construction")
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

print("start session run")
start = time.time()
print(sess.run([pr, delta, step, sum_p, sum_pr]))
end = time.time()
elapse(start, end, "session run")
