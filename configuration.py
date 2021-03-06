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
"""Image-to-text model and training configurations."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


class ModelConfig(object):
    """Wrapper class for model hyperparameters."""

    def __init__(self):
        """Sets the default model hyperparameters."""
        # File pattern of sharded TFRecord file containing SequenceExample
        # protos.
        # Must be provided in training and evaluation modes.
        self.input_file_pattern = None

        self.ft_len = 4096

        # Approximate number of values per input shard. Used to ensure
        # sufficient mixing between shards in training.
        self.values_per_input_shard = 2300
        # Minimum number of shards to keep in the input queue.
        self.input_queue_capacity_factor = 2
        # Number of threads for prefetching SequenceExample protos.
        self.num_input_reader_threads = 2

        # Number of unique words in the vocab (plus 1, for <UNK>). The default
        # value is larger than the expected actual vocab size to allow for
        # differences between tokenizer versions used in preprocessing. There
        # is no harm in using a value greater than the actual vocab size, but
        # using a value less than the actual vocab size will result in an
        # error.
        self.vocab_size = 4000

        # Number of threads for image preprocessing. Should be a multiple of 2.
        self.num_preprocess_threads = 4

        # Batch size.
        self.batch_size = 32

        # Scale used to initialize model variables.
        self.initializer_scale = 0.08

        # LSTM input and output dimensionality, respectively.
        self.embedding_size = self.ft_len
        self.num_lstm_units = 512

        # If < 1.0, the dropout keep probability applied to LSTM variables.
        self.lstm_dropout_keep_prob = 0.7

        self.use_gru = False


class TrainingConfig(object):
    """Wrapper class for training hyperparameters."""

    def __init__(self):
        """Sets the default training hyperparameters."""
        # Number of examples per epoch of training data.
        self.num_examples_per_epoch = 586363

        # Learning rate for the initial phase of training.
        self.initial_learning_rate = 2.0
        self.learning_rate_decay_factor = 0.5
        self.num_epochs_per_decay = 8.0

        # If not None, clip gradients to this value.
        self.clip_gradients = 5.0

        # How many model checkpoints to keep.
        self.max_checkpoints_to_keep = 5
