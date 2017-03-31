import skimage.io as io
#io.use_plugin('matplotlib')
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

IMAGE_HEIGHT = 384
IMAGE_WIDTH = 384
img_dir = 'images'

def get_imagefilename_list(img_dir):
    import os
    filename_list=[]
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.JPG':
                filename_list.append(os.path.join(root, file))
    return filename_list

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def generate_tfrecords(tfrecords_filename,filename_list):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    for img_path in filename_list:
        img = np.array(Image.open(img_path))
        height = img.shape[0]
        width = img.shape[1]
        img_raw = img.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def read_and_decode(filename_queue):
    '''Defining the graph to read and batch images from .tfrecords'''
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
      serialized_example,
      features={
        'height': tf.FixedLenFeature([], tf.int64),
        'width': tf.FixedLenFeature([], tf.int64),
        'image_raw': tf.FixedLenFeature([], tf.string)})

    # Convert from a scalar string tensor
    image = tf.decode_raw(features['image_raw'], tf.uint8)
    height = tf.cast(features['height'], tf.int32)
    width = tf.cast(features['width'], tf.int32)
    image_shape = tf.stack([height, width, 3])
    image = tf.reshape(image, image_shape)

    # Random transformations can be put here: right before you crop images
    # to predefined size.
    resized_image = tf.image.resize_image_with_crop_or_pad(image=image,
                                                           target_height=IMAGE_HEIGHT,
                                                           target_width=IMAGE_WIDTH)
    images = tf.train.shuffle_batch( [resized_image],
                                                 batch_size=2,
                                                 capacity=30,
                                                 num_threads=2,
                                                 min_after_dequeue=10)
    return images

tfrecords_filename = 'test.tfrecords'
if not os.path.exists(tfrecords_filename):
    filename_list = get_imagefilename_list(img_dir)
    generate_tfrecords(tfrecords_filename,filename_list)
filename_queue = tf.train.string_input_producer([tfrecords_filename], num_epochs=10)
# Even when reading in multiple threads, share the filename
# queue.
image = read_and_decode(filename_queue)
print(type(image))
print(image.get_shape())

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(),
                   tf.local_variables_initializer())
with tf.Session()  as sess:

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)  # queue for training

    # Let's read off 3 batches just for example
    #while True:
    for i in range(10):

        img = sess.run([image])
        #print(img[0][0].shape)
        sess.run([loss],feed_dict={x:img,y:label})
        print('current batch')

        # We selected the batch size of two
        # So we should get two image pairs in each batch
        # Let's make sure it is random

        io.imshow(img[0][0])
        #io.show()

        io.imshow(img[0][1])
        #io.show()

    coord.request_stop()
    coord.join(threads)