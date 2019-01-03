from PIL import Image
import tensorflow as tf
import cifar10
import numpy as np
import matplotlib.pyplot as plt
import cv2
import argparse
import pickle


width = 32
height = 32
categories = [ "airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck" ]

parser = argparse.ArgumentParser(description='Image File and Class')
parser.add_argument('filename', type=str, help='image file name')
parser.add_argument('category', type=str, help='what it is')
parser.add_argument('infer_only', type=str, help='Do you want to infer only or not')
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

def min_infer (infer, slice_map):
	# return index of minimum from infer, except for the one's with zero slice_map value
    min_i = -1
    min_j = -1
    min_val = float('inf')

    for i in range(infer.shape[0]):
        for j in range(infer.shape[1]):
            if slice_map[i,j] == 0:
                continue
            if min_val > infer[i,j]:
                min_val = infer[i,j]
                min_i = i
                min_j = j

    return min_i, min_j

def mask_save (image, slice_map, ind, recover=True):
    # saves the image masked with slice_map. If recover option is on, each pixels in slice_map will make
    # 5x5 patch.
    mask_original = image.copy()

    nslice_map = cv2.resize (slice_map, dsize=original.shape[:-1], interpolation=cv2.INTER_AREA)
    
    if recover:
        temp_map = np.zeros (nslice_map.shape)
        temp_map = np.pad (temp_map, [(2,2),(2,2)], mode='constant')
        for i in range(nslice_map.shape[0]):
            for j in range(nslice_map.shape[1]):
                if nslice_map[i,j] == 1:
                    temp_map[i:i+5,j:j+5] += 1
        temp_map = temp_map[2:-2,2:-2]
    else:
        temp_map = nslice_map
    print('saving...  ')
    for i in range(temp_map.shape[0]):
        for j in range(temp_map.shape[1]):
            if temp_map[i,j] == 0:
                mask_original[i,j] = [0,0,0]
                # mask_original[i,j,0] =0
                # mask_original[i,j,1] =0
                # mask_original[i,j,2] =0
            

    im = Image.fromarray(mask_original.astype(np.uint8))
    im.save("./mimages/masked_"+args.category+"_"+str(ind)+".jpeg")
    return mask_original

def slice_save (image, slice_map, ind):
	mask_original = image.copy ()
	nslice_map = cv2.resize (slice_map, dsize=original.shape[:-1], interpolation=cv2.INTER_AREA)

	temp_map = nslice_map

	remain_pix = 0
	avg_rgb = [0,0,0]

	for i in range(temp_map.shape[0]):
	    for j in range(temp_map.shape[1]):
	        if temp_map[i,j] != 0:
	        	avg_rgb += mask_original[i,j]
	        	remain_pix += 1

	avg_rgb /= remain_pix


	for i in range(temp_map.shape[0]):
	    for j in range(temp_map.shape[1]):
	        if temp_map[i,j] == 0:
	            mask_original[i,j] = avg_rgb
	            # mask_original[i,j,0] =0
	            # mask_original[i,j,1] =0
	            # mask_original[i,j,2] =0
	        

	im = Image.fromarray(mask_original.astype(np.uint8))
	im.save("./sliced_images/sliced_"+args.category+"_"+str(ind)+".jpeg")
	return mask_original

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

    original = sess.run(human_image)
    mask_original = original.copy()

    middle = sess.run(front)
    print('middle.shape:        ', middle.shape)
    print('front.get_shape():   ', front.get_shape())
    ori_cost = sess.run (cost, feed_dict= {mimages: middle})

    n_, top_indices = sess.run ([_,top_k_pred], feed_dict= {mimages: middle})
    for key, value in enumerate(top_indices[0]):
        print (categories[value]+"...."+str(value) + ", " + str(n_[0][key]))
    print('normal cost: ',ori_cost)

    if args.infer_only=='y':
        exit (0)


    masked_middle = middle.copy()
    masked_history = []
    pixel_history = []
    cost_history = []

    infer = np.zeros(front.get_shape()[1:3])
    slice_map = np.ones(front.get_shape()[1:3])
    new_infer = np.pad (infer, [(1,1),(1,1)], mode='constant')
    #print('shape of new_infer:  ', new_infer.shape)
    #avg_helper = np.zeros(new_infer.shape)
    avg_helper = np.ones(new_infer.shape)
    #print('infer.shape:         ', infer.shape)
    a = front.get_shape()
    

    for pix in range(infer.shape[0]*infer.shape[1]):
        if (pix+1) % 10 == 0:
            print('Slicing... '+ str(pix+1)+' pixels...')

        for i in range (a[1]):
            #print('yes')
            for j in range (a[2]):
                #print('working on....   ', i, ',', j)
                if slice_map[i,j] == 0:
                    infer[i,j] =0
                    continue

                temp_middle = masked_middle.copy()
                for k in range (a[3]):
                    temp_middle[0,i,j,k] = 0
                #print(masked_middle)
                delta = sess.run(cost, feed_dict={mimages: temp_middle}) - ori_cost
                if delta > 0:
                    infer[i, j] += delta
                
        min_i, min_j = min_infer (infer, slice_map)
        if (min_i == -1) or (min_j == -1):
            print('#### Can\'t slice anymore! ')
            break

        slice_map[min_i,min_j] = 0
        new_middle = masked_middle.copy()
        for k in range(a[3]):
            new_middle[0,min_i,min_j,k] = 0

        #masked_middle = masked_middle * slice_map
        top_indices, new_cost = sess.run([top_k_pred, cost], feed_dict = {mimages: new_middle})
        if top_indices[0][0] != lab:
            slice_map[min_i,min_j] = 1
            print('#### This is not [' +categories[lab]+'] anymore...')
            break
        # if new_cost > 1.2 * ori_cost:
        #     slice_map[min_i,min_j] = 1
        #     print('#### Abrupt change in cost')
        #     break

        if pix % 50 == 0:
            masked_history.append(mask_save(original, slice_map, pix))
            slice_save(original, slice_map, pix)
            # im = Image.fromarray(mask_original.astype(np.uint8))
            # im.save("masked_"+args.category+"_"+str(pix)+".jpeg")
            # masked_history.append(mask_original)
        pixel_history.append((min_i, min_j))
        cost_history.append(new_cost)
        masked_middle = new_middle
        ori_cost = new_cost

    n_, top_indices = sess.run ([_,top_k_pred], feed_dict= {mimages: masked_middle})
    for key, value in enumerate(top_indices[0]):
        print (categories[value]+"...."+str(value) + ", " + str(n_[0][key]))
    print('last cost: ', sess.run(cost, feed_dict={mimages: masked_middle}))

    avg_infer = new_infer / avg_helper
    infer = avg_infer[1:new_infer.shape[0]-1,1:new_infer.shape[1]-1]

    
    
    #fig = plt.figure (figsize=(2,1))
    #new_infer = np.pad (infer, [(5,5),(5,5)], mode='constant')
    am = np.amax(infer)
    if am != 0:
        infer /= am


    #Let's see what mask looks like
    masked_history.append(mask_save(original, slice_map, pix))
    #slice_save(original, slice_map, pix)
            #else:
                #infer[i,j] =0

    #print('image is like...     ', infer.shape)
    # cv2.imshow('mask',original.astype(np.uint8))
    # cv2.waitKey (0)
    # cv2.destroyAllWindows ()



    fig = plt.figure()
    for i, mimg in enumerate(masked_history):
        fig.add_subplot(4,5,i+1)
        plt.imshow(mimg.astype(np.uint8))
    # fig.add_subplot(2,1,1)
    # plt.imshow(original.astype(np.uint8))
    # fig.add_subplot(2,1,2)
    # #plt.imshow(infer, cmap='gray')
    # plt.imshow(mask_original.astype(np.uint8))
    plt.show()

    with open('./masks/'+args.category+'.bin', 'wb') as f:
    	pickle.dump(pixel_history, f)
    	pickle.dump(cost_history, f)


    # infer = np.zeros(front.get_shape())


    # _, top_indices = sess.run([_, top_k_pred])
    # for key, value in enumerate(top_indices[0]):
    #     print (categories[value] + ", " + str(_[0][key]))
