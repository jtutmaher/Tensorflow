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
x_image = tf.reshape(x,[-1,28,28,1])

# DEFINE LAYERS
def weight_variable(shape):
	return tf.Variable(tf.truncated_normal(shape,stddev=0.1))

def bias_variable(shape):
	return tf.Variable(tf.constant(0.1,shape=shape))

def conv2d(x,W):
	return tf.nn.conv2d(x,W,strides=[1,1,1,1],padding='SAME')

def max_pool_2x2(x):
	return tf.nn.max_pool(x,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')

# VARIABLES
W_conv1 = weight_variable([5,5,1,32])
b_conv1 = bias_variable([32])

# FIRST CONVOLUTIONAL LAYER
h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1)+b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# SECOND CONVOLUTIONAL LAYER
W_conv2 = weight_variable([5,5,32,64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2)+b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# DENSELY CONNECTED LAYER
W_fc1 = weight_variable([7*7*64,1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1)+b_fc1)

# DROPOUT
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)

# READOUT
W_fc2 = weight_variable([1024,classes])
b_fc2 = bias_variable([classes])

predict = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)

# LOSS
loss = -tf.reduce_mean(y_*tf.log(tf.clip_by_value(predict,1e-10,1.0)))

# L2 REGULARIZATION
regularizers = tf.nn.l2_loss(W_conv1)+tf.nn.l2_loss(W_conv2)+tf.nn.l2_loss(W_fc1)+tf.nn.l2_loss(W_fc2)
loss += reg*regularizers

# LOSS AND OPTIMIZERS
batch = tf.Variable(0,dtype=tf.float32)
learning_rate = tf.train.exponential_decay(1e-3,batch*batch_size,train_size,0.95)
train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# TEST DATA
correct_prediction = tf.equal(tf.argmax(predict,1),tf.argmax(y_,1))
test_op = tf.reduce_mean(tf.cast(correct_prediction,"float"))

# INITIALIZE VARIABLES
init_op = tf.initialize_all_variables()
sess.run(init_op)

# TRAIN
for step in range(5000):
    batch_xs,batch_ys = mnist.train.next_batch(batch_size)
    pred,l = sess.run([train_op,loss],feed_dict={x:batch_xs,y_:batch_ys,keep_prob:0.5})
    if step%100==0:
        acc = test_op.eval(feed_dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:0.5})
        print "Step %s | Test Accuracy: %s, Loss: %s" %(step,acc,l)