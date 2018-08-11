
# coding: utf-8

# In[7]:


import os
import numpy as np
from datetime import datetime
import tensorflow as tf
import logging


# In[2]:



### -----------------------------------------------------------------
###           INGRADIENTS: atomic elements
### -----------------------------------------------------------------
def build_input_v1():
    """
    Return tensor input
    """
    SENTENCE_MAX_LENGTH = 150
    tf_X = tf.placeholder(dtype=tf.int64, name='tf_X', shape=[None, SENTENCE_MAX_LENGTH])
    tf_y = tf.placeholder(dtype=tf.int64, name='tf_y', shape=[None])
    
    tf_y0 = tf.reduce_sum(tf.cast(tf.equal(tf_y, 0), 'float'))
    tf.summary.scalar(name='y0_count', tensor=tf_y0)
    
    tf_y1 = tf.reduce_sum(tf.cast(tf.equal(tf_y, 1), 'float'))
    tf.summary.scalar(name='y1_count', tensor=tf_y1)
    
    tf_y2 = tf.reduce_sum(tf.cast(tf.equal(tf_y, 2), 'float'))
    tf.summary.scalar(name='y2_count', tensor=tf_y2)
    
    return tf_X, tf_y


def build_input_v2(hyper_parameters={}):
    """
    Return tensor input
    Hyper-parameters: SENTENCE_MAX_LENGTH
    """
    if not 'SENTENCE_MAX_LENGTH' in hyper_parameters:
        hyper_parameters['SENTENCE_MAX_LENGTH'] = 150
    SENTENCE_MAX_LENGTH = hyper_parameters['SENTENCE_MAX_LENGTH']
    
    tf_X = tf.placeholder(dtype=tf.int64, name='tf_X', shape=[None, SENTENCE_MAX_LENGTH])
    tf_y = tf.placeholder(dtype=tf.int64, name='tf_y', shape=[None])
    tf_y0 = tf.reduce_sum(tf.cast(tf.equal(tf_y, 0), 'float'))
    tf_y1 = tf.reduce_sum(tf.cast(tf.equal(tf_y, 1), 'float'))
    tf.summary.scalar(name='y0_count', tensor=tf_y0)
    tf.summary.scalar(name='y1_count', tensor=tf_y1)
    return tf_X, tf_y


def build_inference_v1(tf_X):
    def project(tf_X):
        with tf.device('/cpu:0'), tf.variable_scope('embedding'):
            VOCAB_SIZE = 10000
            EMBEDDING_SIZE = 300

            tf_word_embeddings = tf.get_variable(name='word_embeddings', dtype=tf.float32,
                                              shape=[VOCAB_SIZE, EMBEDDING_SIZE],
                                              initializer=tf.truncated_normal_initializer(stddev=5e-2))
            tf_projected_sentences = tf.nn.embedding_lookup(params=tf_word_embeddings, ids=tf_X)
            return tf_projected_sentences
    
    tf_projected_sens = project(tf_X)
    tf_projected_sens = tf.expand_dims(tf_projected_sens, axis=3)
    
    with tf.variable_scope('convolution_layer'):
        tf_after_conv = tf.layers.conv2d(inputs=tf_projected_sens, filters=10, kernel_size=(5, 5), strides=(2, 2), padding='SAME', name='conv1')
        tf_after_conv = tf.layers.conv2d(inputs=tf_after_conv, filters=20, kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='conv2')
    
    with tf.variable_scope('softmax'):
        tf_flatten = tf.layers.flatten(tf_after_conv)
        tf_logits = tf.layers.dense(inputs=tf_flatten, units=3, activation=tf.nn.relu)
    
    return tf_logits


def build_inference_v2(tf_X, hyper_parameters={}):
    """
    Hyper-parameters: VOCAB_SIZE, EMBEDDING_SIZE
    """
    def project(tf_X):
        with tf.device('/cpu:0'), tf.variable_scope('embedding'):
            if not 'VOCAB_SIZE' in hyper_parameters:
                hyper_parameters['VOCAB_SIZE'] = 10000
            if not 'EMBEDDING_SIZE' in hyper_parameters:
                hyper_parameters['EMBEDDING_SIZE'] = 300
                
            VOCAB_SIZE = hyper_parameters['VOCAB_SIZE']
            EMBEDDING_SIZE = hyper_parameters['EMBEDDING_SIZE']

            tf_word_embeddings = tf.get_variable(name='word_embeddings', dtype=tf.float32,
                                              shape=[VOCAB_SIZE, EMBEDDING_SIZE],
                                              initializer=tf.truncated_normal_initializer(stddev=5e-2))
            tf_projected_sentences = tf.nn.embedding_lookup(params=tf_word_embeddings, ids=tf_X)
            return tf_projected_sentences
    
    tf_projected_sens = project(tf_X)
    tf_projected_sens = tf.expand_dims(tf_projected_sens, axis=3)
    
    with tf.variable_scope('convolution_layer'):
        tf_after_conv = tf.layers.conv2d(inputs=tf_projected_sens, filters=10, kernel_size=(5, 5), strides=(2, 2), padding='SAME', name='conv1')
        tf_after_conv = tf.layers.conv2d(inputs=tf_after_conv, filters=20, kernel_size=(3, 3), strides=(2, 2), padding='SAME', name='conv2')
    
    with tf.variable_scope('softmax'):
        tf_flatten = tf.layers.flatten(tf_after_conv)
        tf_logits = tf.layers.dense(inputs=tf_flatten, units=3, activation=tf.nn.relu)
    
    return tf_logits
    
    
def build_inference_v3(tf_X):
    def project(tf_X):
        with tf.device('/cpu:0'), tf.variable_scope('embedding'):
            VOCAB_SIZE = 10000
            EMBEDDING_SIZE = 300

            tf_word_embeddings = tf.get_variable(name='word_embeddings', dtype=tf.float32,
                                              shape=[VOCAB_SIZE, EMBEDDING_SIZE],
                                              initializer=tf.truncated_normal_initializer(stddev=5e-2))
            tf_projected_sentences = tf.nn.embedding_lookup(params=tf_word_embeddings, ids=tf_X)
            return tf_projected_sentences
    
    tf_projected_sens = project(tf_X)
    tf_projected_sens = tf.expand_dims(tf_projected_sens, axis=3)
    
    with tf.variable_scope('convolution_layer'):
        tf_after_conv = tf.layers.conv2d(inputs=tf_projected_sens, filters=100, kernel_size=(5, 5), strides=(2, 2), padding='SAME', name='conv1')
    
    with tf.variable_scope('softmax'):
        tf_flatten = tf.layers.flatten(tf_after_conv)
        tf_logits = tf.layers.dense(inputs=tf_flatten, units=100, activation=tf.nn.relu)
        tf_logits = tf.layers.dense(inputs=tf_logits, units=3)
    
    return tf_logits
    
    
def build_loss_v1(tf_logits, tf_y):
    tf_losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf_y, logits=tf_logits)
    tf_aggregated_loss = tf.reduce_mean(tf_losses)

    tf.summary.scalar(name='loss', tensor=tf_aggregated_loss)
    return tf_aggregated_loss

def build_optimize_v1(tf_loss):
    """
    Return tensor optimizer and global step
    """
    tf_global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=(), initializer=tf.zeros_initializer())
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.05).minimize(tf_loss, global_step=tf_global_step)
    return optimizer, tf_global_step

def build_optimize_v2(tf_loss):
    """
    Return tensor optimizer and global step
    """
    tf_global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=(), initializer=tf.zeros_initializer())
    opt = tf.train.GradientDescentOptimizer(learning_rate=0.05)
    grads = opt.compute_gradients(tf_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=tf_global_step)
    
    with tf.variable_scope('optimize'):
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
                tf.summary.scalar(var.op.name + '/gradients', tf.nn.l2_loss(grad))
            else:
                logging.warning('Grad is None')
    
    return apply_gradient_op, tf_global_step


def build_optimize_v3(tf_loss):
    """
    Return tensor optimizer and global step
    """
    tf_global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=(), initializer=tf.zeros_initializer())
    opt = tf.train.AdamOptimizer(learning_rate=0.05)
    grads = opt.compute_gradients(tf_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=tf_global_step)
    
    with tf.variable_scope('optimize'):
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
                tf.summary.scalar(var.op.name + '/gradients', tf.nn.l2_loss(grad))
            else:
                logging.warning('Grad is None')
    
    return apply_gradient_op, tf_global_step


def build_optimize_v4(tf_loss):
    """
    Return tensor optimizer and global step
    """
    tf_global_step = tf.get_variable(name='global_step', dtype=tf.int32, shape=(), initializer=tf.zeros_initializer())
    opt = tf.train.AdamOptimizer(learning_rate=0.005)
    grads = opt.compute_gradients(tf_loss)
    apply_gradient_op = opt.apply_gradients(grads, global_step=tf_global_step)
    
    with tf.variable_scope('optimize'):
        # Add histograms for gradients.
        for grad, var in grads:
            if grad is not None:
                tf.summary.histogram(var.op.name + '/gradients', grad)
                tf.summary.scalar(var.op.name + '/gradients', tf.nn.l2_loss(grad))
            else:
                logging.warning('Grad is None')
    
    return apply_gradient_op, tf_global_step


def build_predict_v1(tf_logit):
    """
    Convert from tensor logit to tensor one hot
    """
    tf_predict = tf.argmax(tf_logit, axis=1, name='predict')
    
    tf_predict0 = tf.reduce_sum(tf.cast(tf.equal(tf_predict, 0), 'float'))
    tf.summary.scalar(name='predict0_count', tensor=tf_predict0)
    
    tf_predict1 = tf.reduce_sum(tf.cast(tf.equal(tf_predict, 1), 'float'))
    tf.summary.scalar(name='predict1_count', tensor=tf_predict1)
    
    tf_predict2 = tf.reduce_sum(tf.cast(tf.equal(tf_predict, 2), 'float'))
    tf.summary.scalar(name='predict2_count', tensor=tf_predict2)
    
    # ------------------------
    tf_predict_mean0 = tf.reduce_mean(tf_logit[:, 0])
    tf.summary.scalar(name='predict_mean0', tensor=tf_predict_mean0)
    tf.summary.histogram('predict_0_hist', tf_logit[:, 0])
    
    tf_predict_mean1 = tf.reduce_mean(tf_logit[:, 1])
    tf.summary.scalar(name='predict_mean1', tensor=tf_predict_mean1)
    tf.summary.histogram('predict_1_hist', tf_logit[:, 1])
    
    tf_predict_mean2 = tf.reduce_mean(tf_logit[:, 2])
    tf.summary.scalar(name='predict_mean2', tensor=tf_predict_mean2)
    tf.summary.histogram('predict_2_hist', tf_logit[:, 2])
    
    return tf_predict

def build_accuracy_v1(tf_predict, tf_Y):
    tf_acc = tf.reduce_mean(tf.cast(tf.equal(tf_predict, tf_Y), 'float'), name='accuracy')
    tf.summary.scalar(name='accuracy', tensor=tf_acc)
    return tf_acc

def training_block(graph, tf_X, tf_y, tf_optimizer, tf_global_step, training_generator, test_generator):
    
    with graph.as_default() as gr:
        tf_all_summary = tf.summary.merge_all()
        
        current_dir = os.getcwd()
        experiment_name = datetime.strftime(datetime.now(), '%Y-%m-%dT%H:%M:%S')
        tf_train_writer = tf.summary.FileWriter(logdir=os.path.join(current_dir, 'summary', 'train_' + experiment_name), graph=graph)
        tf_test_writer = tf.summary.FileWriter(logdir=os.path.join(current_dir, 'summary', 'test_' + experiment_name), graph=graph)
        
        with tf.Session().as_default() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(tf.local_variables_initializer())
            SUMMARY_STEP = 10
            EVALUATION_STEP = 10
            for X, y in training_generator:
                feed_dict = {tf_X: X, tf_y: y}
                _, global_step = sess.run([tf_optimizer, tf_global_step], feed_dict=feed_dict)
                
                if global_step % SUMMARY_STEP == 0:
                    logging.debug('Collect summary data at step: %s', global_step)
                    train_summary_data = sess.run(tf_all_summary, feed_dict=feed_dict)
                    tf_train_writer.add_summary(train_summary_data, global_step=global_step)
                    
                if global_step % EVALUATION_STEP == 0:
                    logging.debug('Evaluate at step: %s', global_step)
                    X_test, y_test = next(test_generator)
                    
                    test_summary_data = sess.run(tf_all_summary, feed_dict={
                        tf_X: X_test,
                        tf_y: y_test
                    })
                    tf_test_writer.add_summary(test_summary_data, global_step=global_step)
            tf_train_writer.flush()
            tf_test_writer.flush()
            
def get_training_generator():
    X = np.random.randint(1000, size=(128, 150))
    y = np.random.randint(3, size=(128))
    for i in range(1000):
        yield X, y

