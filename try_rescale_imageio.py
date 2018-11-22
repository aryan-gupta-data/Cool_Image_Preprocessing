import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import skimage
from numpy import matlib as mb
import imageio

# Load training images and labels
x_train_list = np.load('train_images.npy', encoding='bytes')
train_labels = pd.read_csv('train_labels.csv')
x_test_list = np.load('test_images.npy', encoding='bytes')

#imageio.mimwrite('./images', x_train_list)


import numpy as np
import matplotlib as mpl
# mpl.use('Agg')
import matplotlib.pyplot as plt

img = np.ones((100, 200))
img[25:75, 50:150] = 0

fig = plt.figure()
ax = fig.gca()

ax.imshow(img)
ax.axis('tight')

plt.subplots_adjust(0,0,1,1,0,0)

plt.show()

data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
w, h = fig.canvas.get_width_height()
data = data.reshape((h, w, 3))

plt.close()

#--------------------------------------------------------------------------------------------
import numpy as np
import pandas as pd
import random
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import skimage
from numpy import matlib as mb
import imageio
import cv2
import imageio as iio
from skimage import filters
from skimage.color import rgb2gray  # only needed for incorrectly saved images
from skimage.measure import regionprops
from scipy import ndimage

# Load training images and labels
x_train_list = np.load('train_images.npy', encoding='bytes')
train_labels = pd.read_csv('train_labels.csv')
x_test_list = np.load('test_images.npy', encoding='bytes')

x_train = np.zeros((x_train_list[:,1].shape[0], x_train_list[0,1].shape[0]))
x_test = np.zeros((x_test_list[:,1].shape[0], x_test_list[0,1].shape[0]))


# Class information
nb_classes = 31
look_up = {'0': 'sink', '1': 'pear', '2': 'moustache', '3': 'nose', '4': 'skateboard',
           '5': 'penguin', '6': 'peanut', '7': 'skull', '8':'panda', '9': 'paintbrush',
           '10': 'nail', '11': 'apple', '12': 'rifle', '13': 'mug', '14': 'sailboat',
           '15': 'pineapple', '16': 'spoon', '17': 'rabbit', '18': 'shovel', '19': 'rollerskates',
           '20': 'screwdriver', '21': 'scorpion', '22': 'rhinoceros', '23': 'pool', '24':'octagon',
           '25':'pillow', '26': 'parrot', '27': 'squiggle', '28': 'mouth', '29': 'empty', '30': 'pencil'}



# Set up X (training/validation set and test set)
for idx, x in enumerate(x_train_list[:, 1]):
    x_train[idx] = x
    
for idx, x in enumerate(x_test_list[:, 1]):
    x_test[idx] = x

print(x_test.shape, x_train.shape)

# Set up y arrays (both binary and non-binary)
y_categories = []
for target_category in train_labels['Category']:
    y_categories.append(target_category)
    
nb_examples = len(y_categories)
y_train = np.zeros((nb_examples, nb_classes))
y_train_non_bin = np.zeros((nb_examples,))

# Look through the category list; if the category matches an index from the lookup table,
# we'll assign that index to the corresponding location in the training y-vector
for idx, category in enumerate(y_categories):
    for index, image in look_up.items():   
        if image == category:
            y_train[idx, int(index)] = 1
            y_train_non_bin[idx] = int(index)

# Shapes should be nb_examples x nb_features and nb_examples x nb_classes for x and y


x_train /= 255
x_test /= 255

center_ind_list_row = []
center_ind_list_col = []
for i in range(x_train.shape[0]):
    img = (x_train[i].reshape(100,100))
    center_ind_list_col.append((ndimage.measurements.center_of_mass(img)[0]))
    center_ind_list_row.append((ndimage.measurements.center_of_mass(img)[1]))

center_ind_list_row = np.array(center_ind_list_row)
center_ind_list_col = np.array(center_ind_list_col)

for j in range(10):
    img = (x_train[j].reshape(100,100))
    plt.imshow(img)
    plt.axis("off")
    plt.show()
    centre_ind = ndimage.measurements.center_of_mass(img)
    
    size_of_crop = 30
    centre_row = int(round(centre_ind[1]))
    centre_col = int(round(centre_ind[0]))
    #img_crop = img[46-20:46+20,39-20:39+20]
    img_crop = img[(centre_row-size_of_crop):(centre_row+size_of_crop),(centre_col-size_of_crop):(centre_col+size_of_crop)]
    plt.imshow(img_crop)
    plt.axis("off")
    plt.show()
    print ("IMAGE # = ",j)

img = (x_train[10].reshape(100,100))
plt.imshow(img)
plt.axis("off")
plt.show()
size_of_crop = 20
x = 0
y = 0
x_crop = 20
y_crop = 20
number = 0
all_crop_list = []
for row in range(5):
    for col in range(5):
        #centre_ind = ndimage.measurements.center_of_mass(img, index = [1,2])
    #    centre_row = int(round(centre_ind[1]))
    #    centre_col = int(round(centre_ind[0]))
        #img_crop = img[46-20:46+20,39-20:39+20]
        img_crop = img[x:x_crop,y:y_crop]
        all_crop_list.append((np.average(img_crop.reshape((img_crop.shape[0]*img_crop.shape[0]),), axis=0),number))
        plt.imshow(img_crop)
        plt.axis("off")
        plt.show()
        print("size = ", x ,":", x_crop,",",y ,":", y_crop)
        y = y + size_of_crop
        y_crop = y_crop + size_of_crop
        print ('Number = ', number)
        number = number + 1
        
    x = x + size_of_crop
    x_crop = x_crop + size_of_crop
    y = 0
    y_crop = 20

all_crop_list = np.array(all_crop_list)
np.where(all_crop_list == np.max(all_crop_list[:,0]))[0][0]

img = (x_train[10].reshape(100,100))
#img = img[60:80,60:80]
#centre_ind = ndimage.measurements.center_of_mass(img)
size_of_crop = 10
img_crop = img[(40-size_of_crop):(60+size_of_crop),(20-size_of_crop):(40+size_of_crop)]
plt.imshow(img_crop)
plt.axis("off")
plt.show()

img_crop = img[0:20,80:100]
plt.imshow(img_crop)
plt.axis("off")
plt.show()



#ret,thresh = cv2.threshold(img,127,255,0)
#thresh, im_bw = cv2.threshold(img, 127, 255, 0)
#contours,hierarchy = cv2.findContours(im_bw, 1, 2)

x,y,w,h = cv2.boundingRect()
cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

































