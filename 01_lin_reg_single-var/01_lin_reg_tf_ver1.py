import tensorflow as tf
import matplotlib.pyplot as plt

# Hyper-parameters:
n_epoch, lr = 100, 0.01

# Data:
data = [[1], [2], [3]]
target = [[3], [5], [7]]

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# Model:
weight = tf.Variable([[0.1]])
bias = tf.Variable([0.0])

# Training:
output = tf.add(tf.matmul(x, weight), bias)
loss = tf.reduce_mean(tf.square(output - y))

grad_output = 2*(output - y)/tf.cast(tf.shape(y)[0], tf.float32)
grad_weight = tf.matmul(tf.transpose(x), grad_output)
grad_bias = tf.reduce_sum(grad_output, 0)

update_weight = tf.assign(weight, weight - lr*grad_weight)
update_bias = tf.assign(bias, bias - lr*grad_bias)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    loss_train = []
    for n in range(n_epoch):
        output_, loss_ = sess.run([output, loss], feed_dict={x:data, y:target})
        loss_train.append(loss_)
        sess.run([update_weight, update_bias], feed_dict={x:data, y:target})

    print(sess.run(weight), sess.run(bias))

# Evaluation:
fig, (ax1, ax2) = plt.subplots(1,2, figsize=(8,4))
ax1.plot(data, target, 'o')
ax1.plot(data, output_, '--')
ax2.plot(loss_train)
plt.show()
