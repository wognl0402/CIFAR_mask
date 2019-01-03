import pickle
import numpy as np
import tensorflow as tf
from PIL import Image
from matplotlib import pyplot as plt
from matplotlib import animation as anim
import argparse
import cv2

height = 32
width = 32
parser = argparse.ArgumentParser(description='Image File and Class')
parser.add_argument('filename', type=str, help='image file name')
parser.add_argument('original', type=str, help='image file name')

args = parser.parse_args()

with open(args.filename, 'rb') as f:
	pixel_history = pickle.load(f)
	cost_history = pickle.load(f)

xmax = len(cost_history)
ymax = max(cost_history)
ymin = min(cost_history)


fps = 20

class AnimatedGif:
    def __init__(self, size=(640, 480)):
        self.fig = plt.figure()
        self.fig.set_size_inches(size[0] / 100, size[1] / 100)
        ax = self.fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.set_xticks([])
        ax.set_yticks([])
        self.images = []
 
    def add(self, image, label=''):
        plt_im = plt.imshow(image, cmap='Greys', vmin=0, vmax=1, animated=True)
        plt_txt = plt.text(10, 310, label, color='red')
        self.images.append([plt_im, plt_txt])
 
    def save(self, filename):
        animation = anim.ArtistAnimation(self.fig, self.images)
        animation.save(filename, writer='imagemagick', fps=fps)


def mask_save (image, slice_map, recover=True):
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
    #print('saving...  ')
    for i in range(temp_map.shape[0]):
        for j in range(temp_map.shape[1]):
            if temp_map[i,j] == 0:
                mask_original[i,j] = [0,0,0]
                # mask_original[i,j,0] =0
                # mask_original[i,j,1] =0
                # mask_original[i,j,2] =0

    return mask_original


im = Image.open(args.original)
im.convert('RGB')
tmpname = "./tmp.jpeg"
im.save(tmpname, format='JPEG', subsampling=0, quality=100)
input_img = tf.image.decode_jpeg(tf.read_file(tmpname), channels=3)
#input_img = tf.transpose(input_img, (1,2,0))
tf_cast = tf.cast(input_img, tf.float32)
human_image = tf.image.resize_images (tf_cast, [height, width])

with tf.Session() as sess:
	original = sess.run(human_image)

gif = AnimatedGif(size=(320,160))
#gif.add(original, label='0')
images = []
slice_map = np.ones((16,16))

cnt = 0
for i,j in pixel_history:
    cnt += 1
    slice_map[i,j] = 0
    temp = mask_save(original, slice_map)
    temp2 = mask_save(original, slice_map, recover=False)

    temp = cv2.resize(temp, dsize =(160,160), interpolation=cv2.INTER_AREA)
    temp2 = cv2.resize(temp2, dsize = (160,160), interpolation=cv2.INTER_AREA)
    fus = np.concatenate ((temp, temp2), axis=1)

    #if not cnt%4:
    gif.add(fus.astype(np.uint8), label =str(cnt))

    if not cnt%20:
        print('Processing '+ str(cnt)+ 'frames')
    if cnt == len(pixel_history):
    	for k in range(fps*2):
    		gif.add(fus.astype(np.uint8))

print('saving...')
gif.save('animated.gif')
print('done')


plt.figure()
plt.xlim(0,xmax)
plt.ylim(ymin, ymax)
plt.plot(range(len(cost_history)), cost_history)
plt.show()

#plt.waitforbuttonpress(0)