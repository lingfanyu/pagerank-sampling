import tensorflow as tf
from util import load_dataset

d = 0.85
n, indices, values = load_dataset('web.pkl')
values = [d * v for v in values]

# put on CPU first because sparse_reorder does not have a GPU impl...
with tf.device('/device:CPU:0'):
    m = tf.SparseTensor(indices=indices,
                        values=values,
                        dense_shape=[n, n])
    m = tf.sparse_reorder(m)

with tf.device('/device:GPU:0'):
    p = tf.constant(1.0 / n, shape=[n, 1])
    delta = tf.constant(1.0)
    cond = lambda p, delta : delta > 1e-4
    sum_p = tf.reduce_sum(p)
    def body(p, delta):
        last_p = p
        p = tf.sparse_tensor_dense_matmul(m, p) + (1 - d) / n
        p.set_shape([n, 1])
        delta = tf.norm(p - last_p) / tf.norm(last_p)
        return [p, delta]
#    body = lambda p, last_p : \
#        [tf.sparse_tensor_dense_matmul(m, p) + (1 - d) / n, p]
    res = tf.while_loop(cond, body, loop_vars = [p, delta])
            #shape_invariants = [tf.TensorShape([n, 1]), tf.TensorShape([n, 1])])
    pr, delta = res
    sum_pr = tf.reduce_sum(pr)

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
print(sess.run([pr, delta, sum_p, sum_pr]))
