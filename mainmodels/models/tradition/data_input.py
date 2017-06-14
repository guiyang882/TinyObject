from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from models.tradition.config import g_CNNConfig


def read_cifar10(filename_queue):
  class SamplesRecord(object):
    pass

  result = SamplesRecord()

  # Dimensions of the images in the CIFAR-10 dataset.
  # See http://www.cs.toronto.edu/~kriz/cifar.html for a description of the
  # input format.
  label_bytes = 1  # 2 for CIFAR-100
  result.height = g_CNNConfig.image_height
  result.width = g_CNNConfig.image_width
  result.depth = g_CNNConfig.image_depth

  image_bytes = result.height * result.width * result.depth
  # Every record consists of a label followed by the image, with a
  # fixed number of bytes for each.
  total_len = label_bytes + image_bytes
  reader = tf.TFRecordReader()
  result.key, examples = reader.read(filename_queue)
  features = tf.parse_single_example(
      examples,
      features={
          'feature': tf.FixedLenFeature([], tf.string)
      })
  print("#####", features["feature"].get_shape())
  record_bytes = tf.decode_raw(features["feature"], tf.uint8)

  # The first bytes represent the label, which we convert from uint8->int32.
  result.label = tf.cast(
      tf.strided_slice(record_bytes, [-1], [total_len]), tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [height, width, depth].
  result.uint8image = tf.reshape(
      tf.strided_slice(record_bytes, [0], [-1]),
      [result.height, result.width, result.depth])

  return result


def _generate_image_and_label_batch(image, label, min_queue_examples,
                                    batch_size, shuffle):
  """Construct a queued batch of images and labels.

  Args:
    image: 3-D Tensor of [height, width, 3] of type.float32.
    label: 1-D Tensor of type.int32
    min_queue_examples: int32, minimum number of samples to retain
      in the queue that provides of batches of examples.
    batch_size: Number of images per batch.
    shuffle: boolean indicating whether to use a shuffling queue.

  Returns:
    images: Images. 4D tensor of [batch_size, height, width, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  # Create a queue that shuffles the examples, and then
  # read 'batch_size' images + labels from the example queue.
  num_preprocess_threads = 4
  if shuffle:
    images, label_batch = tf.train.shuffle_batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size,
        min_after_dequeue=min_queue_examples)
  else:
    images, label_batch = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=num_preprocess_threads,
        capacity=min_queue_examples + 3 * batch_size)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)

  return images, tf.reshape(label_batch, [batch_size])

def distorted_inputs(filename, batch_size):
  """Construct distorted input for CIFAR training using the Reader ops.

  Args:
    data_dir: Path to the CIFAR-10 data directory.
    batch_size: Number of images per batch.

  Returns:
    images: Images. 4D tensor of [batch_size, IMAGE_SIZE, IMAGE_SIZE, 3] size.
    labels: Labels. 1D tensor of [batch_size] size.
  """
  filenames = [filename]
  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = g_CNNConfig.image_height
  width = g_CNNConfig.image_width
  depth = g_CNNConfig.image_depth

  # Randomly flip the image horizontally.
  float_image = tf.image.random_flip_left_right(reshaped_image)

  # Set the shapes of tensors.
  float_image.set_shape([height, width, depth])
  float_image = tf.divide(float_image, 255.0)
  float_image = tf.subtract(float_image, 0.5)
  # now image has values with zero mean in range [-0.5, 0.5]
  float_image = tf.multiply(float_image, 2.0)

  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(g_CNNConfig.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN *
                           min_fraction_of_examples_in_queue)
  print ('Filling queue with %d CIFAR images before starting to train. '
         'This will take a few minutes.' % min_queue_examples)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=True)

def inputs(test_sample_path, batch_size):
  filenames = [test_sample_path]
  num_examples_per_epoch = g_CNNConfig.NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

  for f in filenames:
    if not tf.gfile.Exists(f):
      raise ValueError('Failed to find file: ' + f)

  # Create a queue that produces the filenames to read.
  filename_queue = tf.train.string_input_producer(filenames)

  # Read examples from files in the filename queue.
  read_input = read_cifar10(filename_queue)
  reshaped_image = tf.cast(read_input.uint8image, tf.float32)

  height = g_CNNConfig.image_height
  width = g_CNNConfig.image_width
  depth = g_CNNConfig.image_depth

  # Image processing for evaluation.

  # Set the shapes of tensors.
  reshaped_image.set_shape([height, width, depth])
  # Randomly flip the image horizontally.
  float_image = tf.image.random_flip_left_right(reshaped_image)
  float_image = tf.divide(float_image, 255.0)
  float_image = tf.subtract(float_image, 0.5)
  # now image has values with zero mean in range [-0.5, 0.5]
  float_image = tf.multiply(float_image, 2.0)

  read_input.label.set_shape([1])

  # Ensure that the random shuffling has good mixing properties.
  min_fraction_of_examples_in_queue = 0.4
  min_queue_examples = int(num_examples_per_epoch *
                           min_fraction_of_examples_in_queue)

  # Generate a batch of images and labels by building up a queue of examples.
  return _generate_image_and_label_batch(float_image, read_input.label,
                                         min_queue_examples, batch_size,
                                         shuffle=False)