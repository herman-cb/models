
"""Contains model definitions for versions of the NIN network.

Usage:
  with slim.arg_scope(nin.nin_arg_scope()):
    outputs, end_points = nin.nin(inputs)

  with slim.arg_scope(nin.nin_arg_scope()):
    outputs, end_points = nin.nin(inputs)

@@nin
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

slim = tf.contrib.slim


def nin_arg_scope(weight_decay=0.0005):
  with slim.arg_scope([slim.conv2d, slim.fully_connected],
                      activation_fn=tf.nn.relu,
                      weights_regularizer=slim.l2_regularizer(weight_decay),
                      biases_initializer=tf.zeros_initializer()):
    with slim.arg_scope([slim.conv2d], padding='SAME') as arg_sc:
      return arg_sc


def nin(inputs,
          num_classes=10,
          is_training=True,
          dropout_keep_prob=0.5,
          spatial_squeeze=True,
          scope='nin',
          fc_conv_padding='SAME',
          global_pool=False):
  with tf.variable_scope(scope, 'nin', [inputs]) as sc:
    end_points_collection = sc.original_name_scope + '_end_points'
    # Collect outputs for conv2d, fully_connected and max_pool2d.
    with slim.arg_scope([slim.conv2d, slim.max_pool2d],
                        outputs_collections=end_points_collection):
      print("Input of nin = {}".format(inputs))
      net = slim.repeat(inputs, 1, slim.conv2d, 192, [5, 5], scope='conv1')
      net = slim.repeat(net, 1, slim.conv2d, 160, [1, 1], scope='conv2')
      net = slim.repeat(net, 1, slim.conv2d, 96, [1, 1], scope='conv3')
      net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool1', padding='SAME')
      print("First layer output = {}".format(net))

      net = slim.repeat(net, 1, slim.conv2d, 192, [5, 5], scope='conv4')
      net = slim.repeat(net, 1, slim.conv2d, 192, [1, 1], scope='conv5')
      net = slim.repeat(net, 1, slim.conv2d, 192, [1, 1], scope='conv6')
      net = slim.max_pool2d(net, [3, 3], stride=2, scope='pool2', padding="SAME")
      
      net = slim.repeat(net, 1, slim.conv2d, 192, [5, 5], scope='conv7')
      net = slim.repeat(net, 1, slim.conv2d, 192, [1, 1], scope='conv8')
      net = slim.repeat(net, 1, slim.conv2d, 10, [1, 1], scope='conv9')
      net = slim.avg_pool2d(net, [8, 8], scope='pool3', stride=1)
      print("Output of nin = {}".format(net))

      end_points = slim.utils.convert_collection_to_dict(end_points_collection)
      if num_classes:
        net = slim.dropout(net, dropout_keep_prob, is_training=is_training,
                           scope='dropout7')
        net = slim.conv2d(net, num_classes, [1, 1],
                          activation_fn=None,
                          normalizer_fn=None,
                          scope='fc8')
        if spatial_squeeze:
          net = tf.squeeze(net, [1, 2], name='fc8/squeezed')
        end_points[sc.name + '/fc8'] = net
      print("Final output of nin = {}".format(net))
      return net, end_points
nin.default_image_size = 32 #TODO


