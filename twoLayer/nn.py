import tensorflow as tf
import numpy as np
import tensorflow.examples.tutorials.mnist.input_data as input_data

# MNIST DATA
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
train_size = mnist.train.labels.shape[0]
batch_size=100
length = 784
intermediate = 1000
classes = 10
reg = 5e-4

# DEFINE SESSION
sess = tf.InteractiveSession()

# INPUTS
x = tf.placeholder("float",shape=[None,length])
y_ = tf.placeholder("float",shape=[None,classes])

# VARIABLES
W1 = tf.Variable(tf.random_normal([length,intermediate]))
b1 = tf.Variable(tf.zeros([intermediate]))
W2 = tf.Variable(tf.random_normal([intermediate,classes]))
b2 = tf.Variable(tf.zeros([classes]))

# MODEL
inter = tf.nn.relu(tf.matmul(x,W1)+b1)
predict = tf.nn.softmax(tf.matmul(inter,W2)+b2)
loss = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(predict,1e-10,1.0)))

# L2 REGULARIZATION
regularizers = tf.nn.l2_loss(W1)+tf.nn.l2_loss(W2)
loss += reg*regularizers

# LOSS AND OPTIMIZERS
batch = tf.Variable(0,dtype=tf.float32)
learning_rate = tf.train.exponential_decay(0.1,batch*batch_size,train_size,0.95)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

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
    	pred,l = sess.run([train_op,loss],feed_dict={x:mnist.test.images,y_:mnist.test.labels})
        acc = test_op.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels})
        print "Accuracy: %s, Loss: %s" %(acc,l)