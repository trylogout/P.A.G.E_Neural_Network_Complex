import cv2
import numpy as np
import sklearn
import mxnet as mx
from PIL import Image
import utils.img_util as imgu
from skimage import transform
from sklearn.decomposition import PCA

def INP(img, bbox=None, landmark=None, **kwargs):
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    if len(str_image_size)>0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size)==1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size)==2
        assert image_size[0]==112
        assert image_size[0]==112 or image_size[1]==96
    
    if landmark is not None:
        assert len(image_size)==2
        src = np.array([
          [30.2946, 51.6963],
          [65.5318, 51.5014],
          [48.0252, 71.7366],
          [33.5493, 92.3655],
          [62.7299, 92.2041] ], dtype=np.float32 )
        if image_size[1]==112:
            src[:,0] += 8.0
        dst = landmark.astype(np.float32)
        tform = transform.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2,:]
        assert len(image_size)==2
        warped = cv2.warpAffine(img,M,(image_size[1],image_size[0]), borderValue = 0.0)
        return warped
    
    if M is None:
        if bbox is None:
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1]*0.0625)
            det[1] = int(img.shape[0]*0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img.shape[1])
        bb[3] = np.minimum(det[3]+margin/2, img.shape[0])
        ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
        if len(image_size)>0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret

def preprocess_input(img, mode):

	if mode == 'emo':
		img = emotion_input_normalization(img)
		return img
	else:
		# blob = image, scale, shape, mean, bool(RGB -> BGR)
		img = cv2.dnn.blobFromImage(img, 1.0, (227,227), (103.939, 116.779, 123.68), True)
		# channel_first -> channel_last
		img = np.rollaxis(img, 1, 4)
	return img

def emotion_input_normalization(single_image):

	single_image = cv2.cvtColor(single_image, cv2.COLOR_BGR2GRAY)
	input_shape 	= (1, 1, 64, 64)
	img 			= Image.fromarray(single_image)
	img 			= img.resize((64, 64), Image.ANTIALIAS)
	T_np 			= np.asarray(img)
	T_np 			= np.resize(T_np, input_shape)
	A, A_pinv 		= imgu.compute_norm_mat(64,64)
	final_image		= imgu.preproc_img(T_np, A,A_pinv) 

	return final_image

def output_normalization(temp):
	temp = int(temp)

	return{
		temp == 0 : 6,
		temp == 1 : 3,
		temp == 2 : 5,
		temp == 3 : 4,
		temp == 4 : 0,
		temp == 5 : 1,
		temp == 6 : 2,
		temp == 7 : 4,
	}[1]

def output_softmax(z):

	assert len(z.shape) == 2
	s = np.max(z, axis=1)
	s = s[:, np.newaxis]
	e_x = np.exp(z - s)
	div = np.sum(e_x, axis=1)
	div = div[:, np.newaxis]

	return e_x / div * 100

def get_input(detector,face_img):
	ret = detector.detect_face(face_img, det_type = 0)
	if ret is None:
		return None
	bbox, points = ret
	if bbox.shape[0]==0:
		return None
	bbox = bbox[0,0:4]
	points = points[0,:].reshape((2,5)).T
	nimg = INP(face_img, bbox, points, image_size='112,112')
	nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
	aligned = np.transpose(nimg, (2,0,1))
	return aligned

def get_feature(model,aligned):
	input_blob = np.expand_dims(aligned, axis=0)
	data = mx.nd.array(input_blob)
	db = mx.io.DataBatch(data=(data,))
	model.forward(db, is_train=False)
	embedding = model.get_outputs()[0].asnumpy()
	embedding = sklearn.preprocessing.normalize(embedding).flatten()
	return embedding