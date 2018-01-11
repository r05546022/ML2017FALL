import numpy as np
import skimage
from skimage import io
import os
import sys
import glob

image_folder_path = sys.argv[1] + '/*.jpg'
image_2_reconstruct_path = sys.argv[1] + '/' + sys.argv[2]


def read_images_in_folder(path):
    image_stack = []
    for img in glob.glob(path):  # All jpeg images
        image_stack.append(io.imread(img))
    return image_stack


def reconstrict(datlist):
    MIN = datlist
    MIN -= np.min(MIN)
    MIN /= np.max(MIN)
    return ((MIN * 255).astype(np.uint8))


image_stack = read_images_in_folder(image_folder_path)
image_flatten = [i.flatten() for i in image_stack]
image_mean_col = [i.mean() for i in np.transpose(image_flatten)]
image_centered = [np.subtract(i, image_mean_col) for i in image_flatten]
image_centered_tr = np.transpose(image_centered)
U, s, V = np.linalg.svd(image_centered_tr, full_matrices=False)
DATA = np.transpose(U)

image_2_reconstruct = io.imread(image_2_reconstruct_path)
rec_original = np.subtract(image_2_reconstruct.flatten(), image_mean_col)
rec_weight = np.inner(DATA[0:4], rec_original)
rec = np.inner(np.transpose(DATA[0:4]), rec_weight)
output = np.reshape(reconstrict(rec + image_mean_col), (600, 600, 3))
io.imsave('reconstruction.jpg', output)

'''
	# Mean face
	imgplot = plt.imshow(np.reshape((np.array(image_mean_col)).astype(np.uint8), (600, 600, 3)))
	# First 4 eigenface
	imgplot = plt.imshow(np.reshape(reconstrict(DATA[0]), (600, 600, 3)))
	plt.show()
    imgplot = plt.imshow(np.reshape(reconstrict(DATA[1]), (600, 600, 3)))
    plt.show()
    imgplot = plt.imshow(np.reshape(reconstrict(DATA[2]), (600, 600, 3)))
    plt.show()
    imgplot = plt.imshow(np.reshape(reconstrict(DATA[3]), (600, 600, 3)))
    plt.show()
    # plot reconstructed output
    imgplot = plt.imshow(output)
    plt.show()

'''
