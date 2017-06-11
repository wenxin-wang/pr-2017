from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


class Vocabulary(object):
    """Vocabulary class for an image-to-text model."""

    def __init__(self,
                 vocab_file,
                 start_word="<S>",
                 end_word="</S>",
                 unk_word="<UNK>"):
        """Initializes the vocabulary.

        Args:
        vocab_file: File containing the vocabulary, where the words are the
        first whitespace-separated token on each line (other tokens are
        ignored) and the word ids are the corresponding line numbers.
        start_word: Special word denoting sentence start.
        end_word: Special word denoting sentence end.
        unk_word: Special word denoting unknown words.
        """
        if not tf.gfile.Exists(vocab_file):
            tf.logging.fatal("Vocab file %s not found.", vocab_file)
        tf.logging.info("Initializing vocabulary from file: %s", vocab_file)

        with tf.gfile.GFile(vocab_file, mode="r") as f:
            reverse_vocab = [line.rsplit(' ', 1)[0] for line in f.readlines()]
        vocab = {x: i for i, x in enumerate(reverse_vocab)}
        assert start_word in vocab
        assert end_word in vocab
        assert unk_word not in vocab
        unk_id = len(reverse_vocab)
        vocab[unk_word] = unk_id
        reverse_vocab.append(unk_word)

        tf.logging.info("Created vocabulary with %d words" % len(vocab))

        self.vocab = vocab  # vocab[word] = id
        self.reverse_vocab = reverse_vocab  # reverse_vocab[id] = word

        # Save special word ids.
        self.start_id = vocab[start_word]
        self.end_id = vocab[end_word]
        self.unk_id = unk_id

    def word_to_id(self, word):
        """Returns the integer word id of a word string."""
        _id = self.vocab.get(word)
        return _id if _id else self.unk_id

    def id_to_word(self, word_id):
        """Returns the word string of an integer word id."""
        if word_id >= len(self.reverse_vocab):
            word_id = self.unk_id
        return self.reverse_vocab[word_id]
