from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from nltk.tokenize.moses import MosesTokenizer
from collections import Counter, defaultdict

import numpy as np
import h5py
from datetime import datetime
import os.path
import sys
import threading

import tensorflow as tf

tf.flags.DEFINE_string("image_ft_file", "../data/image_vgg19_fc1_feature.h5",
                       "Training image directory.")
tf.flags.DEFINE_string("train_captions_file", "../data/train.txt",
                       "Training captions file.")
tf.flags.DEFINE_string("val_captions_file", "../data/valid.txt",
                       "Validation captions file.")

tf.flags.DEFINE_string("output_dir", "../data/records",
                       "Output data directory.")

tf.flags.DEFINE_integer("train_shards", 16,
                        "Number of shards in training TFRecord files.")
tf.flags.DEFINE_integer("val_shards", 2,
                        "Number of shards in validation TFRecord files.")
tf.flags.DEFINE_integer("test_shards", 2,
                        "Number of shards in testing TFRecord files.")

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
tf.flags.DEFINE_string("word_counts_output_file", "word_counts.txt",
                       "Output vocabulary file of word counts.")

tf.flags.DEFINE_integer("num_threads", 4,
                        "Number of threads to preprocess the images.")

FLAGS = tf.flags.FLAGS

tk = MosesTokenizer()


def _ensure_list(value):
    if type(value) is tuple:
        return list(value)
    elif type(value) is np.ndarray:
        return value.tolist()
    elif type(value) is not list:
        value = [value]
    elif not value:
        value = []
    return value


def _int64_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    value = _ensure_list(value)
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def _float_feature(value):
    """Wrapper for inserting an int64 Feature into a SequenceExample proto."""
    value = _ensure_list(value)
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def _to_sequence_example(image, caption, vocab):
    context = tf.train.Features(feature={
        "image/data": _float_feature(image[:]),
        "image/caption_ids":
        _int64_feature([vocab[word] for word in caption]),
    })
    sequence_example = tf.train.SequenceExample(context=context)
    return sequence_example


def _process_image_files(thread_index, ranges, name, images, img_captions,
                         vocab, num_shards):
    # Each thread produces N shards where N = num_shards / num_threads. For
    # instance, if num_shards = 128, and num_threads = 2, then the first thread
    # would produce shards [0, 64).
    num_threads = len(ranges)
    assert not num_shards % num_threads
    num_shards_per_batch = int(num_shards / num_threads)

    shard_ranges = np.linspace(ranges[thread_index][0],
                               ranges[thread_index][1],
                               num_shards_per_batch + 1).astype(int)
    num_images_in_thread = ranges[thread_index][1] - ranges[thread_index][0]

    counter = 0
    for s in range(num_shards_per_batch):
        # Generate a sharded version of the file name,
        # e.g. 'train-00002-of-00010'
        shard = thread_index * num_shards_per_batch + s
        output_filename = "%s-%.5d-of-%.5d" % (name, shard, num_shards)
        output_file = os.path.join(FLAGS.output_dir, output_filename)
        writer = tf.python_io.TFRecordWriter(output_file)

        shard_counter = 0
        images_in_shard = np.arange(
            shard_ranges[s], shard_ranges[s + 1], dtype=int)
        for i in images_in_shard:
            image = images[i]
            if img_captions:
                captions = img_captions[i]
            else:
                captions = [[]]

            for caption in captions:
                sequence_example = _to_sequence_example(image, caption, vocab)
                if sequence_example is None:
                    continue
                writer.write(sequence_example.SerializeToString())
                shard_counter += 1
                counter += 1

                if not counter % 500:
                    print(("%s [thread %d]: Processed %d of %d"
                           " items in thread batch.") %
                          (datetime.now(), thread_index, counter,
                           num_images_in_thread))
                    sys.stdout.flush()
        writer.close()
        print("%s [thread %d]: Wrote %d image-caption pairs to %s" %
              (datetime.now(), thread_index, shard_counter, output_file))
        sys.stdout.flush()
        shard_counter = 0
    print("%s [thread %d]: Wrote %d image-caption pairs to %d shards." %
          (datetime.now(), thread_index, counter, num_shards_per_batch))
    sys.stdout.flush()


def _process_dataset(name, images, img_captions, vocab, num_shards):
    # Break the images into num_threads batches. Batch i is defined as
    # images[ranges[i][0]:ranges[i][1]].
    num_threads = min(num_shards, FLAGS.num_threads)
    spacing = np.linspace(0, len(images), num_threads + 1).astype(np.int)
    ranges = []
    threads = []
    for i in range(len(spacing) - 1):
        ranges.append([spacing[i], spacing[i + 1]])

    # Create a mechanism for monitoring when all threads are finished.
    coord = tf.train.Coordinator()

    # Launch a thread for each batch.
    print("Launching %d threads for spacings: %s" % (num_threads, ranges))
    for thread_index in range(len(ranges)):
        args = (thread_index, ranges, name, images, img_captions, vocab,
                num_shards)
        t = threading.Thread(target=_process_image_files, args=args)
        t.start()
        threads.append(t)

    # Wait for all the threads to terminate.
    coord.join(threads)
    print(
        "%s: Finished processing all %d image-captions pairs in data set '%s'."
        % (datetime.now(), len(images), name))


def _create_vocab(img_captions):
    print("Creating vocabulary.")
    counter = Counter()
    for captions in img_captions:
        for cap in captions:
            counter.update(cap)
    print("Total words:", len(counter))

    # Filter uncommon words and sort by descending count.
    word_counts = [x for x in counter.items() if x[1] >= FLAGS.min_word_count]
    word_counts.sort(key=lambda x: x[1], reverse=True)
    print("Words in vocabulary:", len(word_counts))

    # Write out the word counts file.
    p = os.path.join(FLAGS.output_dir, FLAGS.word_counts_output_file)
    with tf.gfile.FastGFile(p, "w") as f:
        f.write("\n".join(["%s %d" % (w, c) for w, c in word_counts]))
    print("Wrote vocabulary file:", p)

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
                int(line)
                img_caps = []
                caps.append(img_caps)
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

    trn_set, val_set, tst_set = _read_ft_file(FLAGS.image_ft_file)

    _process_dataset("trn", trn_set, trn_caps, vocab, FLAGS.train_shards)
    _process_dataset("val", val_set, val_caps, vocab, FLAGS.val_shards)
    # _process_dataset("tst", tst_set, None, vocab, FLAGS.test_shards)


if __name__ == "__main__":
    tf.app.run()
