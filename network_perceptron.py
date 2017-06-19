from __future__ import print_function
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
from sklearn.metrics import confusion_matrix



tf.reset_default_graph()
# Parameters
learning_rate = 0.001
training_epochs = 1000
batch_size = 100
display_step = training_epochs / 10

# Network Parameters
n_hidden_1 = 128  # 1st layer number of features
n_hidden_2 = 64  # 2nd layer number of features
n_input = 12040  # MNIST data input (img shape: 28*28)
n_classes = 5  # MNIST total classes (0-9 digits

def get_x_data(D):
    A=D.reshape(-1,12040);
    return A
def get_y_data(labels):
    nb_classes = 5
    targets = (labels-1).astype(np.uint32).reshape(-1)
    one_hot_targets = np.eye(nb_classes)[targets]
    return one_hot_targets


def ber(x,y):
    A = confusion_matrix(x, y)
    N = np.shape(A)[0]
    X = 0
    for i in range(N):
        S=np.sum(A[i,:]);
        X=(S+A[i,i])/S
    return X/N


# Create model
def multilayer_perceptron(x, weights, biases, kp):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    #     layer_1 = tf.nn.dropout(layer_1, kp)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    #     layer_2 = tf.nn.dropout(layer_2, kp)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

def train_and_get_accuracy(X,Y,X_t,Y_t):
    # tf Graph input
#     x = tf.placeholder("float", [None, n_input, 86])
    x = tf.placeholder("float", [None, n_input])
    y = tf.placeholder("float", [None, n_classes])
    kp = tf.placeholder(tf.float32)
    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))

    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_classes]))
    }

    # Construct model
    pred = multilayer_perceptron(x, weights, biases, kp)

    # Define loss and optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    cost = cost + 0*sum(reg_losses)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cost)

    # Initializing the variables
    init = tf.global_variables_initializer()

    costs=np.array([0]);

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
                                                          y: batch_y, kp: 1})
            costs=np.append(costs,c)
            # Compute average loss
            avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")
        # np.savetxt('ool.csv', costs, delimiter=',')

        ppp = sess.run([pred], feed_dict={x: get_x_data(X_t),
                                                      y: get_y_data(Y_t), kp: 1})

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))

        net_pred = tf.argmax(pred, 1);

        ppp = sess.run([net_pred], feed_dict={x: get_x_data(X_t),
                                                      y: get_y_data(Y_t), kp: 1})
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        acc=accuracy.eval({x: get_x_data(X_t), y: get_y_data(Y_t), kp: 1});
        b=ber(Y_t-1, (np.asarray(ppp).reshape(-1)).astype(np.int32));
        print("Accuracy:", acc, b)
        return acc,b
