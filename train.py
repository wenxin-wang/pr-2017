# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
"""Train the model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import configuration
from show_and_tell_model import ShowAndTellModel
from show_attend_and_tell_model import ShowAttendAndTellModel

FLAGS = tf.app.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", "",
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("train_dir", "",
                       "Directory for saving and loading model checkpoints.")
tf.flags.DEFINE_integer("number_of_steps", 1000,
                        "Number of training steps.")
tf.flags.DEFINE_integer("log_every_n_steps", 1,
                        "Frequency at which loss and global step are logged.")
tf.flags.DEFINE_integer("attend", 0,
                        "Attend Model")

tf.logging.set_verbosity(tf.logging.INFO)


def main(unused_argv):
    assert FLAGS.input_file_pattern, "--input_file_pattern is required"
    assert FLAGS.train_dir, "--train_dir is required"

    if FLAGS.attend:
        model_config = configuration.AttendModelConfig()
    else:
        model_config = configuration.ModelConfig()
    model_config.input_file_pattern = FLAGS.input_file_pattern
    training_config = configuration.TrainingConfig()

    # Create training directory.
    train_dir = FLAGS.train_dir
    if not tf.gfile.IsDirectory(train_dir):
        tf.logging.info("Creating training directory: %s", train_dir)
        tf.gfile.MakeDirs(train_dir)

    # Build the TensorFlow graph.
    g = tf.Graph()
    with g.as_default():
        # Build the model.
        if FLAGS.attend:
            model = ShowAndTellModel(model_config, mode="train")
        else:
            model = ShowAttendAndTellModel(model_config, mode="train")
        model.build()

        # Set up the training ops.
        train_op = tf.contrib.layers.optimize_loss(
            loss=model.total_loss,
            global_step=model.global_step,
            learning_rate=None,
            optimizer=tf.train.AdamOptimizer())

        # Set up the Saver for saving and restoring model checkpoints.
        saver = tf.train.Saver(
            max_to_keep=training_config.max_checkpoints_to_keep)

    # Run training.
    tf.contrib.slim.learning.train(
        train_op,
        train_dir,
        log_every_n_steps=FLAGS.log_every_n_steps,
        graph=g,
        global_step=model.global_step,
        number_of_steps=FLAGS.number_of_steps,
        init_fn=model.init_fn,
        saver=saver,
        save_interval_secs=300)


if __name__ == "__main__":
    tf.app.run()
