import argparse
import os
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image
import image_process
import scipy.io as scio
import skimage.io as skio
import metric
import h5py
import models

def predict(model_data_path, image_path):

    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    # Read image
    img = Image.open(image_path)
    img = img.resize([width,height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis = 0)

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size)

    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')
        net.load(model_data_path, sess)

        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)

        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        sess.run(init_new_vars_op)

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img}) #<class 'numpy.ndarray'>

        # Plot result
        fig = plt.figure()
        ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
        fig.colorbar(ii)
        plt.show()

        return pred

def batch_predict(model_data_path, image_path_list):

    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1
    rel = 0
    rms = 0
    lg10 = 0
    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size)

    with tf.Session() as sess:

        # Load the converted parameters
        print('Loading the model')
        net.load(model_data_path, sess)

        uninitialized_vars = []
        for var in tf.global_variables():
            try:
                sess.run(var)
            except tf.errors.FailedPreconditionError:
                uninitialized_vars.append(var)
        init_new_vars_op = tf.variables_initializer(uninitialized_vars)
        sess.run(init_new_vars_op)

        #predict the images in image_path_list
        f = h5py.File('test_depth_gt.hdf5','r') # need to modify?????
        i = 0
        for image_path in image_path_list:
            # Read image
            img = Image.open(image_path)
            img = img.resize([width,height], Image.ANTIALIAS)
            img = np.array(img).astype('float32')
            img = np.expand_dims(np.asarray(img), axis = 0)

            # Evalute the network for the given image
            pred = sess.run(net.get_output(), feed_dict={input_node: img}) #<class 'numpy.ndarray'>


            # Plot result
            '''fig = plt.figure()
            ii = plt.imshow(pred[0,:,:,0], interpolation='nearest')
            fig.colorbar(ii)
            plt.show()'''

            # compute the errors
            i +=1
            print(i)
            img_name = os.path.split(image_path)[-1]
            print(img_name)
            dbname = 'depth/' + img_name
            depth_gt = f[dbname]
            rel,rms,lg10 = metric.error_metrics(depth_pred=pred,depth_gt=depth_gt)
            rel += rel
            rms += rms
            lg10 += lg10
        f.close()
        test_images_num = len(image_path_list)
        rel = rel / test_images_num
        rms = rms / test_images_num
        lg10 = lg10 / test_images_num
        print("rel,rms,lg10:",rel,rms,lg10)
        #return pred
def main():
    # Parse arguments
    #parser = argparse.ArgumentParser()
    #parser.add_argument('model_path', help='Converted parameters for the model')
    #parser.add_argument('image_paths', help='Directory of images to predict')
    #args = parser.parse_args()
    data_path = './data/nyu_depth_v2_labeled.mat'
    image_path = './data/test/images'
    model_path = './pretrain/NYU_ResNet-UpProj.npy'

    # Predict the image
    #pred = predict(args.model_path, args.image_paths)
    #image names list

    #print(test_image_list)
    #img = skio.imread(test_image_list[0])
    #skio.imshow(img)
    #skio.show()
    tfrecords_filename = 'test.tfrecords'
    test_image_list = image_process.get_imagefilename_list('./data/test/images')
    if not os.path.exists(tfrecords_filename):
        #image_process.generate_tfrecords(tfrecords_filename,test_image_list)
        image_process.create_record(tfrecords_filename,test_image_list)

    print(test_image_list[0])
    #pred = predict(model_path, test_image_list[0])
    #print(type(pred))
    batch_predict(model_path,test_image_list)
    #os._exit(0)

if __name__ == '__main__':
    main()





