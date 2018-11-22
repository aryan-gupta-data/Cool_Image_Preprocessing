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

#center_ind_list_row = []
#center_ind_list_col = []
#for i in range(x_train.shape[0]):
#    img = (x_train[i].reshape(100,100))
#    center_ind_list_col.append((ndimage.measurements.center_of_mass(img)[0]))
#    center_ind_list_row.append((ndimage.measurements.center_of_mass(img)[1]))
#
#center_ind_list_row = np.array(center_ind_list_row)
#center_ind_list_col = np.array(center_ind_list_col)
#
#for j in range(10):
#    img = (x_train[j].reshape(100,100))
#    plt.imshow(img)
#    plt.axis("off")
#    plt.show()
#    centre_ind = ndimage.measurements.center_of_mass(img)
#    
#    size_of_crop = 30
#    centre_row = int(round(centre_ind[1]))
#    centre_col = int(round(centre_ind[0]))
#    #img_crop = img[46-20:46+20,39-20:39+20]
#    img_crop = img[(centre_row-size_of_crop):(centre_row+size_of_crop),(centre_col-size_of_crop):(centre_col+size_of_crop)]
#    plt.imshow(img_crop)
#    plt.axis("off")
#    plt.show()
#    print ("IMAGE # = ",j)
#i = 125
x_train_crop = []
for i in range(x_train.shape[0]):
    img = (x_train[i].reshape(100,100))
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
            all_crop_list.append((np.average(img_crop.reshape((img_crop.shape[0]*img_crop.shape[0]),), axis=0),number,x,x_crop,y,y_crop))
#            plt.imshow(img_crop)
#            plt.axis("off")
#            plt.show()
#            print("size = ", x ,":", x_crop,",",y ,":", y_crop)
            y = y + size_of_crop
            y_crop = y_crop + size_of_crop
#            print ('Number = ', number)
            number = number + 1
            
        x = x + size_of_crop
        x_crop = x_crop + size_of_crop
        y = 0
        y_crop = 20
    
    all_crop_list = np.array(all_crop_list)
    max_avg_index = np.where(all_crop_list == np.max(all_crop_list[:,0]))[0][0]
#    print (max_avg_index)
    img = (x_train[i].reshape(100,100))
    row1 = int(all_crop_list[max_avg_index][2]-size_of_crop)
    row2 = int(all_crop_list[max_avg_index][3]+size_of_crop)
    col1 = int(all_crop_list[max_avg_index][4]-size_of_crop)
    col2 = int(all_crop_list[max_avg_index][5]+size_of_crop)
    
    size_of_crop = 10
    if np.where(all_crop_list[max_avg_index] == 0)[0].shape[0] == 1: 
        zero_val = np.where(all_crop_list[max_avg_index] == 0)[0][0]
        if zero_val == 2:
            row1 = int(all_crop_list[max_avg_index][2])
            row2 = int(all_crop_list[max_avg_index][3]+(size_of_crop*2))
            col1 = int(all_crop_list[max_avg_index][4]-size_of_crop)
            col2 = int(all_crop_list[max_avg_index][5]+size_of_crop)
        if zero_val == 4:
            col1 = int(all_crop_list[max_avg_index][4])
            col2 = int(all_crop_list[max_avg_index][5]+(size_of_crop*2))
            row1 = int(all_crop_list[max_avg_index][2]-size_of_crop)
            row2 = int(all_crop_list[max_avg_index][3]+size_of_crop)
            
    if np.where(all_crop_list[max_avg_index] == 100)[0].shape[0] == 1: 
        zero_val = np.where(all_crop_list[max_avg_index] == 0)[0][0]
        if zero_val == 3:
            row1 = int(all_crop_list[max_avg_index][2]-(size_of_crop*2))
            row2 = int(all_crop_list[max_avg_index][3])
            col1 = int(all_crop_list[max_avg_index][4]-size_of_crop)
            col2 = int(all_crop_list[max_avg_index][5]+size_of_crop)
        if zero_val == 5:
            row1 = int(all_crop_list[max_avg_index][2]-size_of_crop)
            row2 = int(all_crop_list[max_avg_index][3]+size_of_crop)
            col1 = int(all_crop_list[max_avg_index][4]-(size_of_crop*2))
            col2 = int(all_crop_list[max_avg_index][5])
            
    img_crop = img[row1:row2,col1:col2]
#    print (i,img_crop.shape)
    x_train_crop.append((img_crop.reshape(img_crop.shape[0]*img_crop.shape[0],)))
#    plt.imshow(img_crop)
#    plt.axis("off")
#    plt.show()

x_train_crop = np.array(x_train_crop)


plt.imshow(x_train_crop[2].reshape(40,40))
plt.axis("off")
plt.show()
