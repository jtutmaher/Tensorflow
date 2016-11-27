import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

# MNIST DATA
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
batch_size=100
length = 784
classes = 10

# DEFINE SESSION
sess = tf.InteractiveSession()

# INPUTS
x = tf.placeholder("float",shape=[None,length])
y_ = tf.placeholder("float",shape=[None,classes])

# VARIABLES
W1 = tf.Variable(tf.zeros([length,classes]))
b1 = tf.Variable(tf.zeros([classes]))

# MODEL
predict = tf.nn.softmax(tf.matmul(x,W1)+b1)
loss = -tf.reduce_mean(y_*tf.log(predict))

# LOSS AND OPTIMIZERS
train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

# TEST DATA
correct_prediction = tf.equal(tf.argmax(predict,1),tf.argmax(y_,1))
test_op = tf.reduce_mean(tf.cast(correct_prediction,"float"))

# INITIALIZE VARIABLES
init_op = tf.initialize_all_variables()
sess.run(init_op)

# TRAIN
for step in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(batch_size)
    sess.run(train_op,feed_dict={x:batch_xs,y_:batch_ys})
    if step%100==0:
        print test_op.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels})