from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(1.0, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x,W,strides=[1,2,2,1], padding='SAME')


def apply_max_pool(x,kernel_size,stride_size):
    return tf.nn.max_pool(x, ksize=[1, kernel_size, kernel_size, 1],
                          strides=[1, stride_size, stride_size, 1], padding='SAME')

def get_x_data(D):
    A=D.reshape(-1,6020,2)
    return A


def get_y_data(labels):
    nb_classes = 5
    targets = (labels-1).astype(np.uint32).reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets


def construct_net(x,weights,biases):
    # x is of size [batch, width, num channels]
    graph = tf.nn.conv1d(x,weights['c1'], stride=1, padding='SAME',use_cudnn_on_gpu=False);
    graph = tf.add(graph,biases['b1']);
    graph = tf.nn.relu(graph);

    shape = graph.get_shape().as_list()
    features = tf.reshape(graph, [-1, shape[1] * shape[2]])

    print(features.get_shape());

    graph = tf.matmul(features, weights['n1']);
    print(graph.get_shape());
    graph = tf.add(graph, biases['b2']);

    return graph;


def train_and_get_accuracy(X,Y,X_t,Y_t):
    num_labels = 5
    num_channels = 1

    X=X.astype(np.float32)
    X_t=X_t.astype(np.float32)

    learning_rate = 0.01
    training_epochs = 100
    display_step=1

    x = tf.placeholder("float", shape=[None,6020, 2])
    y = tf.placeholder("float", shape=[None,num_labels])

    # 20*5

    # (5,100)*(100,1)=(5,1)

    weights = {
        'c1': tf.Variable(tf.random_normal([301, 2, 5])),
        'n1': tf.Variable(tf.random_normal([30100,5]))

    }
    biases = {
        'b1': tf.Variable(tf.random_normal([6020, 5])),
        'b2': tf.Variable(tf.random_normal([5]))
    }


    pred = construct_net(x,weights,biases);

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cost = cost + 0 * sum(reg_losses)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
    #     optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    costs=np.array([]);

    # Launch the graph
    with tf.Session() as sess:
        sess.run(init)

        batch_x = get_x_data(X)
        batch_y = get_y_data(Y)
        # Training cycle
        for epoch in range(training_epochs):
            avg_cost = 0.
            total_batch = 1
            # Run optimization op (backprop) and cost op (to get loss value)
            _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,
                                                          y: batch_y})

            costs=np.append(costs,c)
            # Compute average loss
            avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch + 1), "cost=", \
                      "{:.9f}".format(avg_cost))
        print("Optimization Finished!")
        np.savetxt('oolc.csv', costs, delimiter=',')
        0/0

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        acc = accuracy.eval({x: get_x_data(X_t), y: get_y_data(Y_t)});
        print("Accuracy:", acc)
        return acc
