# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Converts Aerobotics data to TFRecord file format with Example protos."""

import math
import os
import random
import sys
import augmented_build_data
import tensorflow as tf
from PIL import Image
import numpy as np

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('image_folder',
                           './AeroDevKit/Aerobotics/JPEGImages',
                           'Folder containing images.')

tf.app.flags.DEFINE_string(
    'semantic_segmentation_folder',
    './AeroDevKit/Aerobotics/SegmentationClass',
    'Folder containing semantic segmentation annotations.')

tf.app.flags.DEFINE_string(
    'list_folder',
    './AeroDevKit/Aerobotics/ImageSets',
    'Folder containing lists for training and validation')

tf.app.flags.DEFINE_string(
    'output_dir',
    './tfrecord',
    'Path to save converted SSTable of TensorFlow examples.')

tf.app.flags.DEFINE_string(
    'run_options',
    'images',
    'Tells the script if it should read np arrays or images.')


_NUM_SHARDS = 2


def _convert_dataset(dataset_split,dataset_dir):
    """Converts the Aerobotics dataset into into tfrecord format.

    Args:
      dataset_split: Dataset split (e.g., train, val).
      dataset_dir: Dir in which the dataset locates.
      dataset_label_dir: Dir in which the annotations locates.

    Raises:
      RuntimeError: If loaded image and label have different shape.
    """

    img_names = tf.gfile.Glob(os.path.join(dataset_dir, '1.png'))

    ''' Opens image and saves the number of dimensions to read in '''
    image = Image.open(img_names[0])
    np_image = np.asarray(image)
    img_channels = np_image.shape[2]

    dataset = os.path.basename(dataset_split)[:-4]
    sys.stdout.write('Processing ' + dataset)
    filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
    num_images = len(filenames)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = augmented_build_data.ImageReader('png', channels=img_channels)

    label_reader = augmented_build_data.ImageReader('png', channels=1)

    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(
            FLAGS.output_dir,
            '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
              sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                  i + 1, len(filenames), shard_id))
              sys.stdout.flush()
              # Read the image.
              image_filename = os.path.join(
                  FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
              image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
              height, width = image_reader.read_image_dims(image_data)
              # Read the semantic segmentation annotation.
              seg_filename = os.path.join(
                  FLAGS.semantic_segmentation_folder,
                  filenames[i] + '.' + FLAGS.label_format)
              seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
              seg_height, seg_width = label_reader.read_image_dims(seg_data)
              if height != seg_height or width != seg_width:
                raise RuntimeError('Shape mismatched between image and label.')
              # Convert to tf example.
              example = augmented_build_data.image_seg_to_tfexample(
                  image_data, filenames[i], height, width, seg_data, img_channels)
              tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()

def _convert_dataset_gif(dataset_split,dataset_dir):
    """Converts the Aerobotics dataset into into tfrecord format.

    Args:
      dataset_split: Dataset split (e.g., train, val).
      dataset_dir: Dir in which the dataset locates.
      dataset_label_dir: Dir in which the annotations locates.

    Raises:
      RuntimeError: If loaded image and label have different shape.
    """

    dataset = os.path.basename(dataset_split)[:-4]
    sys.stdout.write('Processing ' + dataset)
    filenames = [x.strip('\n') for x in open(dataset_split, 'r')]
    num_images = len(filenames)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    image_reader = augmented_build_data.ImageReader('gif', 3)

    label_reader = augmented_build_data.ImageReader('png', channels=1)

    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(
            FLAGS.output_dir,
            '%s-%05d-of-%05d.tfrecord' % (dataset, shard_id, _NUM_SHARDS))
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
              sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                  i + 1, len(filenames), shard_id))
              sys.stdout.flush()
              # Read the image.
              image_filename = os.path.join(
                  FLAGS.image_folder, filenames[i] + '.' + FLAGS.image_format)
              image_data = tf.gfile.FastGFile(image_filename, 'rb').read()
              channels, height, width, frames = image_reader.read_image_dims_gif(image_data)
              # Read the semantic segmentation annotation.
              seg_filename = os.path.join(
                  FLAGS.semantic_segmentation_folder,
                  filenames[i] + '.' + FLAGS.label_format)
              seg_data = tf.gfile.FastGFile(seg_filename, 'rb').read()
              seg_height, seg_width = label_reader.read_image_dims(seg_data)
              if height != seg_height or width != seg_width:
                raise RuntimeError('Shape mismatched between image and label.')
              # Convert to tf example.
              example = augmented_build_data.image_seg_to_tfexample_gif(image_data, filenames[i], height, width, seg_data, channels, frames)
              tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()

def main(unused_argv):
    if FLAGS.run_options == 'images':
        dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_folder, '*.txt'))
        for dataset_split in dataset_splits:
          _convert_dataset(dataset_split,FLAGS.image_folder)
    elif FLAGS.run_options == 'gif':
        dataset_splits = tf.gfile.Glob(os.path.join(FLAGS.list_folder, '*.txt'))
        for dataset_split in dataset_splits:
          _convert_dataset_gif(dataset_split,FLAGS.image_folder)



if __name__ == '__main__':
  tf.app.run()
