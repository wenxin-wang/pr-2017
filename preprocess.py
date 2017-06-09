from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nltk.tokenize.moses import MosesTokenizer
from collections import Counter, defaultdict

import h5py
from datetime import datetime
import os.path
import sys

import tensorflow as tf


tf.flags.DEFINE_string("image_ft1_file", "../data/image_vgg19_fc1_feature.h5",
                       "Training image directory.")
tf.flags.DEFINE_string("image_ft2_file", "../data/image_vgg19_fc2_feature.h5",
                       "Training image directory.")
tf.flags.DEFINE_string("image_cnn_file",
                       "../data/image_vgg19_block5_pool_feature.h5",
                       "Training image directory.")

tf.flags.DEFINE_string("train_captions_file", "../data/train.txt",
                       "Training captions file.")
tf.flags.DEFINE_string("val_captions_file", "../data/valid.txt",
                       "Validation captions file.")

tf.flags.DEFINE_string("output_dir", "../data/records",
                       "Output data directory.")

tf.flags.DEFINE_string("start_word", "<S>",
                       "Special word added to the beginning of each sentence.")
tf.flags.DEFINE_string("end_word", "</S>",
                       "Special word added to the end of each sentence.")
tf.flags.DEFINE_string("unknown_word", "<UNK>",
                       "Special word meaning 'unknown'.")
tf.flags.DEFINE_integer(
    "min_word_count", 4,
    "The minimum number of occurrences of each word in the "
    "training set for inclusion in the vocabulary.")
tf.flags.DEFINE_string("word_counts_output_file",
                       "../data/records/word_counts.txt",
                       "Output vocabulary file of word counts.")

FLAGS = tf.flags.FLAGS

tk = MosesTokenizer()


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _int64_feature_list(values):
    """Wrapper for inserting an int64 FeatureList into a SequenceExample proto.
    """
    return tf.train.FeatureList(feature=[_int64_feature(v) for v in values])


def _to_sequence_example(img_id, caption, vocab):
    context = tf.train.Features(feature={
        "image/image_id":
        _int64_feature(img_id)
    })

    caption_ids = [vocab[word] for word in caption]
    feature_lists = tf.train.FeatureLists(feature_list={
        "image/caption_ids":
        _int64_feature_list(caption_ids)
    })
    sequence_example = tf.train.SequenceExample(
        context=context, feature_lists=feature_lists)

    return sequence_example


def _process_caption_files(name, img_captions, vocab):
    output_filename = "%s-captions.tfr"
    output_file = os.path.join(FLAGS.output_dir, output_filename)
    writer = tf.python_io.TFRecordWriter(output_file)
    counter = 0
    for img_id, captions in img_captions:
        for caption in captions:
            sequence_example = _to_sequence_example(img_id, caption, vocab)
            if sequence_example is not None:
                writer.write(sequence_example.SerializeToString())
                counter += 1

            if not counter % 1000:
                print("%s :Processed %d captoins." % (datetime.now(),
                                                            counter))
                sys.stdout.flush()

    writer.close()
    print("%s : Wrote %d captions to %s" % (datetime.now(), counter,
                                            output_file))
    sys.stdout.flush()


def _process_dataset(name, images, img_captions, vocab):
    for image, (_, captions) in zip(images, img_captions):
        pass


def _create_vocab(img_captions):
    print("Creating vocabulary.")
    counter = Counter()
    for _, captions in img_captions:
        for cap in captions:
            counter.update(cap)
    print("Total words:", len(counter))

    # Filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))

    # Write out the word counts file.
    with tf.gfile.FastGFile(FLAGS.word_counts_output_file, "w") as f:
        f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    print("Wrote vocabulary file:", FLAGS.word_counts_output_file)

    unk_id = len(word_counts)
    vocab = defaultdict(lambda: unk_id)
    for i, (x, _) in enumerate(word_counts):
        vocab[x] = i
    return vocab


def _process_caption(caption):
    """Processes a caption string into a list of tonenized words.

  Args:
    caption: A string caption.

  Returns:
    A list of strings; the tokenized caption.
  """
    tokenized_caption = [FLAGS.start_word]
    tokenized_caption.extend(tk.tokenize(caption.lower()))
    tokenized_caption.append(FLAGS.end_word)
    return tokenized_caption


def _read_captions(ifname):
    caps = []
    img_caps = None
    with open(ifname, 'r') as f:
        for line in f:
            line = line.rstrip('\n')
            try:
                i = int(line)
                img_caps = []
                caps.append((i, img_caps))
            except ValueError:
                img_caps.append(_process_caption(line))
    return caps


def _read_ft_file(ifname):
    h5f = h5py.File(ifname, 'r')
    trn_set = h5f['train_set'][:, :]
    val_set = h5f['validation_set'][:, :]
    tst_set = h5f['test_set'][:, :]
    h5f.close()
    return trn_set, val_set, tst_set


def main(unused_argv):
    if not tf.gfile.IsDirectory(FLAGS.output_dir):
        tf.gfile.MakeDirs(FLAGS.output_dir)

    # Load caption files.
    trn_caps = _read_captions(FLAGS.train_captions_file)
    val_caps = _read_captions(FLAGS.val_captions_file)

    # Create vocabulary from the training captions.
    vocab = _create_vocab(trn_caps)
    _process_caption_files("trn", trn_caps, vocab)
    _process_caption_files("val", val_caps, vocab)

    #trn_set, val_set, tst_set = _read_ft_file(FLAGS.image_ft_file)

    #_process_dataset("trn", trn_set, trn_caps, vocab, FLAGS.train_shards)


if __name__ == "__main__":
    tf.app.run()
