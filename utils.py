import numpy as np
import scipy.misc as misc
import cPickle as pickle
import cv2
import os

image_height = 300
image_width = 200
image_channel = 3

def load_data(folder):
    """ Load all images from a given folder
    Parameter:
        The directory path to the data folder.

    Return:
        dataset: a 4D array in the form [index, image_height, image_width,
                    image_channel]
        label_index: a 1D array, contains labels (in number) corresponding to
                    the images in the dataset.
        label_tag: contains tags in plain text for the labels.
    """
    label_tag = os.listdir(folder)
    label_index = []

    # Iterate over all subfolders:
    for i in range(len(label_tag)):
        tag = label_tag[i]
        subpath = os.path.join(folder, tag)
        image_list = os.listdir(subpath)

        #Iterate over all images in the subfolder:
        for image in image_list:
            image_path = os.path.join(subpath, image)
            try:
                image_array = misc.imread(image_path)
                image_array = format_image(image_array)
                image_array = normalize_image(image_array)
                if 'dataset' in locals():
                    dataset = np.concatenate( (dataset, [image_array]), axis=0 )
                else:
                    dataset = np.ndarray(
                            shape=(1, image_height, image_width, image_channel),
                            dtype=np.float32)
                label_index.append(i+1) # The label ranging from 1 to num_categories
            except IOError as e:
                print 'Could not read:', e, ', skipping.'
    db_size = len(label_index) # The data size
    label_index = np.array(label_index, dtype=np.float32)

    # Randomize the dataset and the label:
    choice = np.arange(db_size)
    np.random.shuffle(choice)
    label_index = label_index[choice]
    dataset = dataset[choice, :, :, :]

    return dataset, label_index, label_tag
    
def format_image(img):
    """ Resize an image to a fixed size, may change the image ratio, and make
    it RGB if the image is gray """
    img = misc.imresize(img, (image_height, image_width))
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return img

def normalize_image(img):
    return (img.astype(float) - 255.0 / 2) / 255.0

def next_batch(X, Y, batch_size):
    """ Load a subset of data from dataset """
    db_size = X.shape[0]
    assert db_size >= batch_size
    choice = np.arange(db_size)
    np.random.shuffle(choice)
    choice = choice[0:batch_size]
#    choice = np.random.choice(db_size, batch_size)
    batchX = X[choice, :, :, :]
    batchY = Y[choice]
    return batchX, batchY

def one_hot(label, max_label):
    """ Generate one-hot rows for given label """
    for i in range(label.shape[0]):
        current = (label[i] == np.arange(1, max_label+1))
        if not 'result' in locals():
            result = np.array([current])
        else:
            result = np.concatenate((result, [current]), axis=0)
    return result.astype(np.float32)