import tensorflow as tf
from tensorflow.contrib.rnn.python.ops import core_rnn_cell
from tensorflow.contrib.rnn.python.ops import core_rnn_cell_impl
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import tanh, sigmoid

_linear = core_rnn_cell_impl._linear
_checked_scope = core_rnn_cell_impl._checked_scope


class AttentionCell(core_rnn_cell.RNNCell):
    def __init__(self,
                 num_units,
                 num_attns,
                 attn_size,
                 forget_bias=1.0,
                 activation=tanh,
                 reuse=None):
        self._num_units = num_units
        self._num_attns = num_attns
        self._attn_size = attn_size
        self._forget_bias = forget_bias
        self._activation = activation
        self._reuse = reuse

    @property
    def state_size(self):
        return (self._num_units, self._num_units,
                self._num_attns * self._attn_size)

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, scope=None):
        """Long short-term memory cell (LSTM)."""
        with _checked_scope(
                self, scope or "basic_lstm_cell", reuse=self._reuse):
            c, h, all_attns = state

            with vs.variable_scope("attn_selection"):
                logits = _linear([all_attns, h], self._num_attns, True)
            softmax = tf.nn.softmax(logits)
            all_attns_mat = tf.reshape(all_attns, [-1, self._num_attns,
                                                   self._attn_size])
            new_attn = tf.reduce_sum(softmax * all_attns_mat, 1)

            with vs.variable_scope("attn_gate"):
                beta = _linear([h], self._attn_size, True)

            z = sigmoid(beta) * new_attn

            concat = _linear([inputs, h, z], 4 * self._num_units, True)

            # i = input_gate, j = new_input, f = forget_gate, o = output_gate
            i, j, f, o = tf.split(
                value=concat, num_or_size_splits=4, axis=1)

            new_c = (c * sigmoid(f + self._forget_bias) +
                     sigmoid(i) * self._activation(j))
            new_h = self._activation(new_c) * sigmoid(o)

            return new_h, (new_c, new_h, all_attns)
