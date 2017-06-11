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
r"""Generate captions for images using default beam search parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import tensorflow as tf
import h5py

import configuration
import inference_wrapper
from inference_utils import caption_generator
from inference_utils import vocabulary

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("checkpoint_path", "",
                       "Model checkpoint file or directory containing a "
                       "model checkpoint file.")
tf.flags.DEFINE_string("vocab_file", "",
                       "Text file containing the vocabulary.")
tf.flags.DEFINE_string("input_file", "",
                       "h5file of image files.")
tf.flags.DEFINE_string("dataset", "test_set",
                       "dataset to use")
tf.flags.DEFINE_integer("attend", 0,
                                "Attend Model")


tf.logging.set_verbosity(tf.logging.INFO)


def main(_):
    # Build the inference graph.
    g = tf.Graph()
    with g.as_default():
        model = inference_wrapper.InferenceWrapper(FLAGS.attend)
        if FLAGS.attend:
            model_config = configuration.AttendModelConfig()
        else:
            model_config = configuration.ModelConfig()
        restore_fn = model.build_graph_from_config(model_config,
                                                   FLAGS.checkpoint_path)
    g.finalize()

    # Create the vocabulary.
    vocab = vocabulary.Vocabulary(FLAGS.vocab_file)

    with tf.Session(graph=g) as sess:
        # Load the model from checkpoint.
        restore_fn(sess)

        h5f = h5py.File(FLAGS.input_file)
        images = h5f[FLAGS.dataset][:]
        images = images.reshape((len(images), -1))
        tf.logging.info("Running caption generation on %d images in %s",
                        images.shape[0], FLAGS.input_file)
        # Prepare the caption generator. Here we are implicitly using the
        # default beam search parameters. See caption_generator.py for a
        # description of the available beam search parameters.
        generator = caption_generator.CaptionGenerator(model, vocab)

        for _id, image in enumerate(images):
            captions = generator.beam_search(sess, image)
            print(_id)
            for caption in captions:
                # Ignore begin and end words.
                sentence = [
                    vocab.id_to_word(w) for w in caption.sentence[1:-1]
                ]
                sentence = " ".join(sentence)
                print("%f@@ %s" % (math.exp(caption.logprob), sentence))
        h5f.close()


if __name__ == "__main__":
    tf.app.run()
