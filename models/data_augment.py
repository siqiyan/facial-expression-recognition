import cv2
import numpy as np
from random import random

from time import sleep
import matplotlib.pyplot as plt


def process(img):
	

	out = np.copy(img)

	if int(0.5+ random()):
		out = random_noise(out)

	if int(0.5+ random()):
		out = random_crop(out)

	if int(0.5+ random()):		
		out = random_pos(out)

	if int(0.5+ random()):	
		out = random_rotation(out)

	if int(0.5+ random()):
		out = horizontal_flip(out)

	if int(0.5+ random()):
		out = jitter_contrast(out)

	if int(0.5+ random()):
		out = random_colour(out)

	return out
	

def random_pos(img):
	(h,w) = img.shape[:2]
	x0 = -10 + random()*(w/5)
	y0 = -10 + random()*(h/5)

	translation_matrix = np.float32([ [1,0,x0], [0,1,y0] ])
	out = cv2.warpAffine(img, translation_matrix, (w, h))

	return out

'Takes RGB image and changes the colour randomly'
def random_colour(img):
	d = int(random()*2.999)
	alpha = random()*1.5

	out = np.copy(img)

	out[:,:,d] = img[:,:,d]*alpha

	return out

'Takes RGB image and changes the rotation randomly'
def random_rotation(img):
	(h, w) = img.shape[:2]
	center = (w / 2, h / 2)
	theta = -20 + random()*40
	M = cv2.getRotationMatrix2D(center, theta, 1.0)
	out = cv2.warpAffine(img, M, (w, h))
	return out

'Takes RGB image and returns a random crop of the image'
def random_crop(img):
	(h, w) = img.shape[:2]
	h_1_4 = h/4.0
	w_1_4 = w/4.0


	x0 = random()*w_1_4
	y0 = random()*h_1_4

	out = img[x0:x0+(3)*w_1_4, y0:y0+(3)*h_1_4]

	out = cv2.resize(out, (h, w), interpolation = cv2.INTER_AREA)

	return out

'Takes RGB image and returns a horizontally flipped image'
def horizontal_flip(img):
	return cv2.flip(img, 1)


def jitter_contrast(img):

	maxIntensity = 255.0 # depends on dtype of image data

	# Parameters for manipulating image data
	alpha = 1 + random()*0.5
	beta = 1 + random()*0.5
	phi = 0.2 + random()*2

	out = np.copy(img)
	out[:,:,0] = (maxIntensity/alpha)*(out[:,:,0]/(maxIntensity/beta))**phi
	out[:,:,1] = (maxIntensity/alpha)*(out[:,:,1]/(maxIntensity/beta))**phi
	out[:,:,2] = (maxIntensity/alpha)*(out[:,:,2]/(maxIntensity/beta))**phi	

	return out

def random_noise(img):
	noise = np.zeros(img.shape, np.uint8)
	# cv2.randn(noise,(0),(99),)
	alpha = 0.1*random()
	cv2.randn(noise, np.zeros(3), np.ones(3)*255*alpha)
 
	out = img + noise

	return out

if __name__ == '__main__':
	img = cv2.imread('face.jpg')
	cv2.imshow('Image', img)

	while (True):
		
		img2 = process(img)
		

		cv2.imshow('Image Proc',img2)
		

		if cv2.waitKey(1) & 0xFF == ord('q'):
			pass

		sleep(0.5)





