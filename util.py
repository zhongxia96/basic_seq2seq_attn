# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains some utility functions"""

import tensorflow as tf
import time
import os
FLAGS = tf.app.flags.FLAGS


def linear(args, output_size, bias, scope='Linear'):
    """Linear map: sum_i(args[i]*W[i])
       args: a 2D Tensor or a list of Tensor, Tensor shape: (batch_size, n)
       return : (batch_size, output_size)
    """
    if not isinstance(args, (list, tuple)):
        args= [args]
    total_args_size = 0
    shapes = [a.get_shape().as_list() for a in args]
    for shape in shapes:
        total_args_size += shape[1]
    with tf.variable_scope(scope):
        W = tf.get_variable("W", [total_args_size, output_size])
        res = tf.matmul(tf.concat(args, axis=1), W)
        if bias:
            bias_term = tf.get_variable("bias_term", [output_size])
            res += bias_term
    return res

def load_ckpt(saver, sess, ckpt_dir='train'):
    while True:
        try:
            ckpt_dir = os.path.join(FLAGS.log_root, ckpt_dir)
            ckpt_state = tf.train.get_checkpoint_state(ckpt_dir)
            print 'loading checkpoint from ', ckpt_state.model_checkpoint_path
            saver.restore(sess, ckpt_state.model_checkpoint_path)
            return ckpt_state.model_checkpoint_path
        except:
            tf.logging.info('failed to load checkpoint from %s. sleeping for %i secs', ckpt_dir, 10)
            time.sleep(10)

def get_config():
  """Returns config for tf.session"""
  config = tf.ConfigProto(allow_soft_placement=True)
  config.gpu_options.allow_growth=True
  return config
