import tensorflow as tf

# Set hyper-parameters:
n_epoch, lr = 100, 0.01

# Load data: y = 2 x + 1
data = [[1], [2], [3]]
target = [[3], [5], [7]]

x = tf.placeholder(tf.float32, [None, 1])
y = tf.placeholder(tf.float32, [None, 1])

# Setup a model:
weight = tf.Variable([[0.1]])
bias = tf.Variable([0.0])

# Forward propagation:
output = tf.add(tf.matmul(x, weight), bias)
loss = tf.reduce_mean(tf.square(output - y))

# Backward propagation:
grad_output = 2*(output - y)/tf.cast(tf.shape(y)[0], tf.float32)
grad_weight = tf.matmul(tf.transpose(x), grad_output)
grad_bias = tf.reduce_sum(grad_output, 0)

# Update model parameters:
update_weight = weight.assign(weight - lr*grad_weight)
update_bias = bias.assign(bias - lr*grad_bias)

# To prevent CUDA_ERROR_OUT_OF_MEMORY:
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())

    # Train the model:
    for epoch in range(n_epoch):
        loss_ = sess.run(loss, feed_dict={x:data, y:target})
        sess.run([update_weight, update_bias], feed_dict={x:data, y:target})

    # Evaluate the trained model:
    print("Epoch[%3d] >> Loss = %f" % (n_epoch, loss_))
    print(weight.eval(), bias.eval())