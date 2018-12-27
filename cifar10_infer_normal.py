from PIL import Image
import tensorflow as tf
import cifar10
import numpy as np
import matplotlib.pyplot as plt
#import cv2

# from mxnet import gluon, nd, image
# from mxnet.gluon.data.vision import transforms
# from gluoncv import utils
#from gluoncv.model_zoo import get_model

#from tensorflow.models.image.cifar10 import cifar10
width = 32
height = 32
categories = [ "airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck" ]

filename = "./sample_img/cat.jpg" # absolute path to input image

#img = image.imread(filename)
#img= transform_fn(img)
#tf_cast = tf.cast (img, tf.float32)

im = Image.open(filename)

tmpname = "./tmp.jpeg"
im.save(tmpname, format='JPEG', subsampling=0, quality=100)
input_img = tf.image.decode_jpeg(tf.read_file(tmpname), channels=3)
#input_img = tf.transpose(input_img, (1,2,0))
tf_cast = tf.cast(input_img, tf.float32)
human_image = tf.image.resize_images (tf_cast, [height, width])
float_image = tf.image.per_image_standardization (human_image)
#float_image = tf.image.resize_image_with_crop_or_pad(tf_cast, height, width)
images = tf.expand_dims(float_image, 0)
logits = cifar10.inference(images)
#front = cifar10.inference_sliced_front (images)
#mimages = tf.placeholder("float", front.get_shape())
#logits = cifar10.inference_sliced_back (mimages)
_, top_k_pred = tf.nn.top_k(logits, k=5)
init_op = tf.initialize_all_variables()
label = [2]
labels = tf.cast (label, tf.int64)

cost = cifar10.loss(logits, labels)

with tf.Session() as sess:
 # Restore variables from training checkpoint.
    sess.run(init_op)
    variable_averages = tf.train.ExponentialMovingAverage(
        cifar10.MOVING_AVERAGE_DECAY)
    variables_to_restore = variable_averages.variables_to_restore()
    saver = tf.train.Saver(variables_to_restore)
    ckpt = tf.train.get_checkpoint_state('./cifar10_train/')
    if ckpt and ckpt.model_checkpoint_path:
        print("ckpt.model_checkpoint_path ", ckpt.model_checkpoint_path)
        saver.restore(sess, ckpt.model_checkpoint_path)
    else:
        print('No checkpoint file found')
        exit(0)
    #sess.run(init_op)

    
    ori_cost = sess.run (cost)

    _, top_indices = sess.run ([_,top_k_pred])
    for key, value in enumerate(top_indices[0]):
        print (categories[value]+"...."+str(value) + ", " + str(_[0][key]))
    print('normal cost: ',ori_cost)
    # infer = np.zeros(front.get_shape())
    original = sess.run(human_image)[:,:,:]

    plt.figure()
    plt.imshow(original.astype(np.uint8))
    plt.show()

    # _, top_indices = sess.run([_, top_k_pred])
    # for key, value in enumerate(top_indices[0]):
    #     print (categories[value] + ", " + str(_[0][key]))
