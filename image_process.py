import skimage.io as io
import scipy.io as scio
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import h5py

IMAGE_HEIGHT = 228
IMAGE_WIDTH = 304
img_dir = 'images'
NYU_LABELED_PATH = './data/nyu_depth_v2_labeled.mat'
SPLIT_TRAIN_TEST_PATH = './data/splits.mat'

def get_imagefilename_list(img_dir):
    import os
    filename_list=[]
    for root, dirs, files in os.walk(img_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.jpg':
                filename_list.append(os.path.join(root, file))
    return filename_list

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def generate_tfrecords(tfrecords_filename,filename_list):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)

    i = 0
    for img_path in filename_list:
        i = i + 1 # just for test
        print(i)
        #img = np.array(Image.open(img_path))
        img = Image.open(img_path)
        #height = img.shape[0]
        height = np.array(img).shape[0]
        #width = img.shape[1]
        width = np.array(img).shape[1]
        #img_raw = img.tostring()
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={
            'height': _int64_feature(height),
            'width': _int64_feature(width),
            'image_raw': _bytes_feature(img_raw)}))
        writer.write(example.SerializeToString())
    writer.close()

def create_record(tfrecords_filename,filename_list):
    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    # writer = tf.python_io.TFRecordWriter("data/data_new_4_3/" + rotated + data_class + ".tfrecords")
    i = 1
    for img_name in filename_list:
        i = i+1
        print(i)
        img = Image.open(img_name)
        #img = img.resize([304,228])
        img_raw = img.tobytes()  # 将图片转化为原生bytes
        example = tf.train.Example(
            features=tf.train.Features(feature={
                'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))
            }))
        writer.write(example.SerializeToString())
    writer.close()


def read_and_decode(filename_queue):  #？？different in training and test case
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
                                                 batch_size=16,
                                                 capacity=1,
                                                 num_threads=2,
                                                 min_after_dequeue=0)
    return images

def parse_NYU_v2_data(nyu_file='./data/nyu_depth_v2_labeled.mat'):
    '''parse the NYU v2 depth dataset
    :param:
    '''
    split = scio.loadmat(SPLIT_TRAIN_TEST_PATH)
    #split.keys():trainNdxs,testNdxs
    train_index = split['trainNdxs']
    train_index = np.concatenate(train_index) # concatenate to 1 dim array
    test_index = split['testNdxs']
    test_index = np.concatenate(test_index)

    #split the NYU v2 labeled dataset
    NYU_v2 = h5py.File(NYU_LABELED_PATH)
    images = NYU_v2['images'] # 1449*3*640*480
    #images = np.transpose(images,[0,3,2,1]) # 1449*480*640*3

    depths = NYU_v2['depths'] # 1449*640*480
    #depths = np.transpose(depths,[0,2,1]) # 1449*480*640
    img_names = NYU_v2['rawDepthFilenames'] #1*1449

    #parse training images
    depth_train_gt = []
    depth_test_gt = []
    train_set_num = len(train_index)
    test_set_num = len(test_index)
    ff=0
    '''for i in range(train_set_num):
        train_img_index = train_index[i] - 1 # index from 1 to 1449, but from 0 to 1448 for array
        #print(train_img_index)
        train_img = images[train_img_index] # 3*640*480
        train_img = np.transpose(train_img,[2,1,0]) # 480*640*3
        train_img = Image.fromarray(train_img)
        tmp = img_names[0][train_img_index]
        obj = NYU_v2[tmp]
        train_img_name = ''.join(chr(k) for k in obj[:])
        train_img_name = train_img_name.replace('/','-')
        train_img_name = train_img_name + '.jpg'
        train_img_name = os.path.join(os.getcwd(),'data/train/images',train_img_name)
        #print(train_img_name)
        train_img.save(train_img_name)

        #save the depth groundtruth
        train_img_depth = depths[train_img_index] # 640*480
        train_img_depth = np.transpose(train_img_depth,[1,0]) #480*640
        depth_train_gt.append(train_img_depth)
        ff=ff+1
        print(ff)'''
    #depth_train_gt_filename = os.path.join(os.getcwd(),'data/train/depth','depth_train_gt.mat')
    #scio.savemat(depth_train_gt_filename,mdict={'depth':np.array(depth_train_gt),'index':train_index})

    # parse test images
    f = h5py.File("test_depth_gt.hdf5", "w")
    for  i in range(test_set_num):
        test_img_index = test_index[i] - 1 # index from 1 to 1449, but from 0 to 1448 for array
        #print(test_img_index)
        test_img = images[test_img_index] # 3*640*480
        test_img = np.transpose(test_img,[2,1,0]) # 480*640*3
        test_img = Image.fromarray(test_img)
        tmp = img_names[0][test_img_index]
        obj = NYU_v2[tmp]
        test_img_name = ''.join(chr(k) for k in obj[:])
        test_img_name = test_img_name.replace('/','-')
        test_img_name = test_img_name + '.jpg'
        dataset_name = 'depth/'+test_img_name
        test_img_name = os.path.join(os.getcwd(),'data/test/images',test_img_name)
        #print(test_img_name)
        #test_img.save(test_img_name)

        #save the depth groundtruth
        test_img_depth = depths[test_img_index] # 640*480
        test_img_depth = np.transpose(test_img_depth,[1,0]) #480*640
        f.create_dataset(dataset_name,dtype='f',data=test_img_depth)

    f.close()

#parse_NYU_v2_data()
def create_h5_dataset():
    '''f = h5py.File("mytestfile.hdf5", "w")
    #dset1 = f.create_dataset("mydataset", (100,), dtype='i')
    #f.create_dataset()
    dset2 = f.create_dataset("depth/image1",dtype='f',data=np.random.randn(10,10))
    dset3 = f.create_dataset("depth/image2",(10,10),dtype='f')

    f.close()'''

    f = h5py.File("mytestfile.hdf5","r")
    image1 = f['depth/image1']
    image1= np.array(image1)
    print(image1)
    print(image1.shape)
    f.close()

#create_h5_dataset()
'''import os
flist = get_imagefilename_list('./data/test/images')
fname = os.path.split(flist[0])
print(flist[0])
print(fname)'''