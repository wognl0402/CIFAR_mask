from PIL import Image
import tensorflow as tf
import cifar10
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse


width = 32
height = 32
categories = [ "airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck" ]

parser = argparse.ArgumentParser(description='Image File and Class')
parser.add_argument('filename', type=str, help='image file name')
parser.add_argument('category', type=str, help='what it is')
args = parser.parse_args()

lab = 0
for cat in categories:
    if cat == args.category:
        break
    lab += 1
if lab == 10:
    print('###### category does not match from the list ######')
    exit(0)
#filename = "./sample_img/deer.png" # absolute path to input image
filename = args.filename
#img = image.imread(filename)
#img= transform_fn(img)
#tf_cast = tf.cast (img, tf.float32)

im = Image.open(filename)
im.convert('RGB')
tmpname = "./tmp.jpeg"
im.save(tmpname, format='JPEG', subsampling=0, quality=100)
input_img = tf.image.decode_jpeg(tf.read_file(tmpname), channels=3)
#input_img = tf.transpose(input_img, (1,2,0))
tf_cast = tf.cast(input_img, tf.float32)
human_image = tf.image.resize_images (tf_cast, [height, width])
float_image = tf.image.per_image_standardization (human_image)
#float_image = tf.image.resize_image_with_crop_or_pad(tf_cast, height, width)
images = tf.expand_dims(float_image, 0)
#logits = cifar10.inference(images)
front = cifar10.inference_sliced_front_v1 (images)
mimages = tf.placeholder("float", front.get_shape())
logits = cifar10.inference_sliced_back_v1 (mimages, images)
_, top_k_pred = tf.nn.top_k(logits, k=5)
init_op = tf.initialize_all_variables()
label = [lab]
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

    middle = sess.run(front)
    print('middle.shape:        ', middle.shape)
    print('front.get_shape():   ', front.get_shape())
    ori_cost = sess.run (cost, feed_dict= {mimages: middle})

    _, top_indices = sess.run ([_,top_k_pred], feed_dict= {mimages: middle})
    for key, value in enumerate(top_indices[0]):
        print (categories[value]+"...."+str(value) + ", " + str(_[0][key]))
    print('normal cost: ',ori_cost)

    masked_middle = middle.copy()
    infer = np.zeros(front.get_shape()[1:3])
    new_infer = np.pad (infer, [(1,1),(1,1)], mode='constant')
    print('shape of new_infer:  ', new_infer.shape)
    #avg_helper = np.zeros(new_infer.shape)
    avg_helper = np.ones(new_infer.shape)
    print('infer.shape:         ', infer.shape)
    a = front.get_shape()
    for i in range (a[1]):
        #print('yes')
        for j in range (a[2]):
            #print('working on....   ', i, ',', j)
            temp_middle = middle.copy()
            for k in range (a[3]):
                temp_middle[0,i,j,k] = 0
            #print(masked_middle)
            delta = sess.run(cost, feed_dict={mimages: temp_middle}) - ori_cost
            if delta > 0:
                new_infer[i+1, j+1] += delta
                #new_infer[i:i+3,j:j+3] += delta
            #avg_helper[i:i+3,j:j+3] += 1

    avg_infer = new_infer / avg_helper
    infer = avg_infer[1:new_infer.shape[0]-1,1:new_infer.shape[1]-1]

    original = sess.run(human_image)

    
    #fig = plt.figure (figsize=(2,1))
    #new_infer = np.pad (infer, [(5,5),(5,5)], mode='constant')
    am = np.amax(infer)
    if am != 0:
        infer /= am


    #Let's see what mask looks like
    # infer = cv2.resize (infer, dsize=original.shape[:-1])
    # for i in range(infer.shape[0]):
    #     for j in range(infer.shape[1]):
    #         if infer[i,j] > 0:
    #             infer[i,j] =1
    #         else:
    #             infer[i,j] =0

    #print('image is like...     ', infer.shape)
    # cv2.imshow('mask',original.astype(np.uint8))
    # cv2.waitKey (0)
    # cv2.destroyAllWindows ()



    fig = plt.figure()
    fig.add_subplot(2,1,1)
    plt.imshow(original.astype(np.uint8))
    fig.add_subplot(2,1,2)
    plt.imshow(infer, cmap='gray')
    plt.show()


    # infer = np.zeros(front.get_shape())


    # _, top_indices = sess.run([_, top_k_pred])
    # for key, value in enumerate(top_indices[0]):
    #     print (categories[value] + ", " + str(_[0][key]))
