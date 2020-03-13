import numpy as np
import random as rnd
from PIL import Image
from scipy import ndimage

def compute_norm_mat(base_width, base_height): 
	x      = np.arange(base_width)
	y      = np.arange(base_height)
	X, Y   = np.meshgrid(x, y)
	X      = X.flatten()
	Y      = Y.flatten() 
	A      = np.array([X*0+1, X, Y]).T 
	A_pinv = np.linalg.pinv(A)
	return A, A_pinv

def preproc_img(img, A, A_pinv):
	img_flat = img.flatten()
	img_hist = np.bincount(np.int64(img_flat), minlength = 256)
	cdf = img_hist.cumsum() 
	cdf = cdf * (2.0 / cdf[-1]) - 1.0
	img_eq = cdf[np.int64(img_flat)] 
	diff = img_eq - np.dot(A, np.dot(A_pinv, img_eq))
	std = np.sqrt(np.dot(diff,diff)/diff.size)
	if std > 1e-6: 
		diff = diff/std
	return diff.reshape(img.shape)