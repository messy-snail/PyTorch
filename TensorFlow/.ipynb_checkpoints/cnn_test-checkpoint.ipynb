{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "C:\\Users\\ISL\\Anaconda3\\lib\\site-packages\\h5py\\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.\n",
      "  from ._conv import register_converters as _register_converters\n"
     ]
    }
   ],
   "source": [
    "# Lab 11 MNIST and Deep learning CNN\n",
    "import tensorflow as tf\n",
    "import random\n",
    "# import matplotlib.pyplot as plt\n",
    "\n",
    "from tensorflow.examples.tutorials.mnist import input_data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From <ipython-input-2-dc4a0f0089fa>:3: read_data_sets (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use alternatives such as official/mnist/dataset.py from tensorflow/models.\n",
      "WARNING:tensorflow:From C:\\Users\\ISL\\Anaconda3\\lib\\site-packages\\tensorflow\\contrib\\learn\\python\\learn\\datasets\\mnist.py:260: maybe_download (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please write your own downloading logic.\n",
      "WARNING:tensorflow:From C:\\Users\\ISL\\Anaconda3\\lib\\site-packages\\tensorflow\\contrib\\learn\\python\\learn\\datasets\\base.py:252: _internal_retry.<locals>.wrap.<locals>.wrapped_fn (from tensorflow.contrib.learn.python.learn.datasets.base) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use urllib or similar directly.\n",
      "Successfully downloaded train-images-idx3-ubyte.gz 9912422 bytes.\n",
      "WARNING:tensorflow:From C:\\Users\\ISL\\Anaconda3\\lib\\site-packages\\tensorflow\\contrib\\learn\\python\\learn\\datasets\\mnist.py:262: extract_images (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use tf.data to implement this functionality.\n",
      "Extracting MNIST_data/train-images-idx3-ubyte.gz\n",
      "Successfully downloaded train-labels-idx1-ubyte.gz 28881 bytes.\n",
      "WARNING:tensorflow:From C:\\Users\\ISL\\Anaconda3\\lib\\site-packages\\tensorflow\\contrib\\learn\\python\\learn\\datasets\\mnist.py:267: extract_labels (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use tf.data to implement this functionality.\n",
      "Extracting MNIST_data/train-labels-idx1-ubyte.gz\n",
      "WARNING:tensorflow:From C:\\Users\\ISL\\Anaconda3\\lib\\site-packages\\tensorflow\\contrib\\learn\\python\\learn\\datasets\\mnist.py:110: dense_to_one_hot (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use tf.one_hot on tensors.\n",
      "Successfully downloaded t10k-images-idx3-ubyte.gz 1648877 bytes.\n",
      "Extracting MNIST_data/t10k-images-idx3-ubyte.gz\n",
      "Successfully downloaded t10k-labels-idx1-ubyte.gz 4542 bytes.\n",
      "Extracting MNIST_data/t10k-labels-idx1-ubyte.gz\n",
      "WARNING:tensorflow:From C:\\Users\\ISL\\Anaconda3\\lib\\site-packages\\tensorflow\\contrib\\learn\\python\\learn\\datasets\\mnist.py:290: DataSet.__init__ (from tensorflow.contrib.learn.python.learn.datasets.mnist) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "Please use alternatives such as official/mnist/dataset.py from tensorflow/models.\n"
     ]
    }
   ],
   "source": [
    "tf.set_random_seed(777)  # reproducibility\n",
    "\n",
    "mnist = input_data.read_data_sets(\"MNIST_data/\", one_hot=True)\n",
    "# Check out https://www.tensorflow.org/get_started/mnist/beginners for\n",
    "# more information about the mnist dataset\n",
    "\n",
    "# hyper parameters\n",
    "learning_rate = 0.001\n",
    "training_epochs = 15\n",
    "batch_size = 100"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "# dropout (keep_prob) rate  0.7~0.5 on training, but should be 1 for testing\n",
    "keep_prob = tf.placeholder(tf.float32)\n",
    "\n",
    "# input place holders\n",
    "X = tf.placeholder(tf.float32, [None, 784])\n",
    "X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)\n",
    "Y = tf.placeholder(tf.float32, [None, 10])\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "# L1 ImgIn shape=(?, 28, 28, 1)\n",
    "W1 = tf.Variable(tf.random_normal([3, 3, 1, 32], stddev=0.01))\n",
    "#    Conv     -> (?, 28, 28, 32)\n",
    "#    Pool     -> (?, 14, 14, 32)\n",
    "L1 = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')\n",
    "L1 = tf.nn.relu(L1)\n",
    "L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1],\n",
    "                    strides=[1, 2, 2, 1], padding='SAME')\n",
    "L1 = tf.nn.dropout(L1, keep_prob=keep_prob)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "metadata": {},
   "outputs": [],
   "source": [
    "# L2 ImgIn shape=(?, 14, 14, 32)\n",
    "W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))\n",
    "#    Conv      ->(?, 14, 14, 64)\n",
    "#    Pool      ->(?, 7, 7, 64)\n",
    "L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')\n",
    "L2 = tf.nn.relu(L2)\n",
    "L2 = tf.nn.max_pool(L2, ksize=[1, 2, 2, 1],\n",
    "                    strides=[1, 2, 2, 1], padding='SAME')\n",
    "L2 = tf.nn.dropout(L2, keep_prob=keep_prob)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "metadata": {},
   "outputs": [],
   "source": [
    "# L3 ImgIn shape=(?, 7, 7, 64)\n",
    "W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))\n",
    "#    Conv      ->(?, 7, 7, 128)\n",
    "#    Pool      ->(?, 4, 4, 128)\n",
    "#    Reshape   ->(?, 4 * 4 * 128) # Flatten them for FC\n",
    "L3 = tf.nn.conv2d(L2, W3, strides=[1, 1, 1, 1], padding='SAME')\n",
    "L3 = tf.nn.relu(L3)\n",
    "L3 = tf.nn.max_pool(L3, ksize=[1, 2, 2, 1], strides=[\n",
    "                    1, 2, 2, 1], padding='SAME')\n",
    "L3 = tf.nn.dropout(L3, keep_prob=keep_prob)\n",
    "L3_flat = tf.reshape(L3, [-1, 128 * 4 * 4])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "metadata": {},
   "outputs": [],
   "source": [
    "# L4 FC 4x4x128 inputs -> 625 outputs\n",
    "W4 = tf.get_variable(\"W4\", shape=[128 * 4 * 4, 625],\n",
    "                     initializer=tf.contrib.layers.xavier_initializer())\n",
    "b4 = tf.Variable(tf.random_normal([625]))\n",
    "L4 = tf.nn.relu(tf.matmul(L3_flat, W4) + b4)\n",
    "L4 = tf.nn.dropout(L4, keep_prob=keep_prob)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "# L5 Final FC 625 inputs -> 10 outputs\n",
    "W5 = tf.get_variable(\"W5\", shape=[625, 10],\n",
    "                     initializer=tf.contrib.layers.xavier_initializer())\n",
    "b5 = tf.Variable(tf.random_normal([10]))\n",
    "logits = tf.matmul(L4, W5) + b5"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "WARNING:tensorflow:From <ipython-input-9-1315abe87b3f>:4: softmax_cross_entropy_with_logits (from tensorflow.python.ops.nn_ops) is deprecated and will be removed in a future version.\n",
      "Instructions for updating:\n",
      "\n",
      "Future major versions of TensorFlow will allow gradients to flow\n",
      "into the labels input on backprop by default.\n",
      "\n",
      "See @{tf.nn.softmax_cross_entropy_with_logits_v2}.\n",
      "\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# define cost/loss & optimizer\n",
    "cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(\n",
    "    logits=logits, labels=Y))\n",
    "optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Learning started. It takes sometime.\n",
      "Epoch: 0001 cost = 0.371361985\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# initialize\n",
    "sess = tf.Session()\n",
    "sess.run(tf.global_variables_initializer())\n",
    "\n",
    "# train my model\n",
    "print('Learning started. It takes sometime.')\n",
    "for epoch in range(training_epochs):\n",
    "    avg_cost = 0\n",
    "    total_batch = int(mnist.train.num_examples / batch_size)\n",
    "\n",
    "    for i in range(total_batch):\n",
    "        batch_xs, batch_ys = mnist.train.next_batch(batch_size)\n",
    "        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}\n",
    "        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)\n",
    "        avg_cost += c / total_batch\n",
    "\n",
    "    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))\n",
    "\n",
    "print('Learning Finished!')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'tf' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-1-6b957640f20e>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m()\u001b[0m\n\u001b[0;32m      4\u001b[0m \u001b[1;31m# if you have a OOM error, please refer to lab-11-X-mnist_deep_cnn_low_memory.py\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      5\u001b[0m \u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m----> 6\u001b[1;33m \u001b[0mcorrect_prediction\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mequal\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0margmax\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mlogits\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0margmax\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mY\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;36m1\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m      7\u001b[0m \u001b[0maccuracy\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0mtf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mreduce_mean\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mtf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mcast\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mcorrect_prediction\u001b[0m\u001b[1;33m,\u001b[0m \u001b[0mtf\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mfloat32\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      8\u001b[0m print('Accuracy:', sess.run(accuracy, feed_dict={\n",
      "\u001b[1;31mNameError\u001b[0m: name 'tf' is not defined"
     ]
    }
   ],
   "source": [
    "\n",
    "# Test model and check accuracy\n",
    "\n",
    "# if you have a OOM error, please refer to lab-11-X-mnist_deep_cnn_low_memory.py\n",
    "\n",
    "correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))\n",
    "accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))\n",
    "print('Accuracy:', sess.run(accuracy, feed_dict={\n",
    "      X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))\n",
    "\n",
    "# Get one and predict\n",
    "r = random.randint(0, mnist.test.num_examples - 1)\n",
    "print(\"Label: \", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))\n",
    "print(\"Prediction: \", sess.run(\n",
    "    tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Test model and check accuracy\n",
    "\n",
    "# if you have a OOM error, please refer to lab-11-X-mnist_deep_cnn_low_memory.py\n",
    "\n",
    "correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))\n",
    "accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))\n",
    "print('Accuracy:', sess.run(accuracy, feed_dict={\n",
    "      X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))\n",
    "\n",
    "# Get one and predict\n",
    "r = random.randint(0, mnist.test.num_examples - 1)\n",
    "print(\"Label: \", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))\n",
    "print(\"Prediction: \", sess.run(\n",
    "    tf.argmax(logits, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))\n",
    "\n",
    "# plt.imshow(mnist.test.images[r:r + 1].\n",
    "#           reshape(28, 28), cmap='Greys', interpolation='nearest')\n",
    "# plt.show()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
