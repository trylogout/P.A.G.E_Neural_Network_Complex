import os
import cv2
import mxnet as mx
import numpy as np
import cntk as ct
from scipy import misc
import matplotlib.pyplot as plt
from utils.detector import MtcnnDetector
from utils.lib_connection import lib_conn
import utils.preprocess as PTA

def person_recognize(img1, img2):
	ctx = mx.gpu(0)
	det_threshold = [0.6,0.7,0.8]
	mtcnn_path = os.path.join(os.path.dirname(__file__), 'det_mtcnn')
	detector = MtcnnDetector(model_folder=mtcnn_path, 
						ctx=ctx, 
						num_worker=1, 
						accurate_landmark = True, 
						threshold=det_threshold)
	model = lib_conn('PR')
	pre1 = PTA.get_input(detector,img1)
	out1 = PTA.get_feature(model,pre1)
	pre2 = PTA.get_input(detector,img2)
	# plt.imshow(np.transpose(pre2,(1,2,0)))
	# plt.show()
	out2 = PTA.get_feature(model,pre2)
	#dist = np.sum(np.square(out1-out2))
	sim = np.dot(out1, out2.T)
	return sim*100
	
def age_gender_reconize(simple_image):
	age_list        = ['0-2','4-6','8-12','15-20','25-32','38-43','48-53','60-100']
	gender_list     = ['Male','Female']
	age_model, gen_model = lib_conn('AGR')
	img = PTA.preprocess_input(simple_image, "age")
	img = img.copy(order='C')
	age_predictions = np.asarray((age_model(np.float32(img)))[0])
	age_predictions = PTA.output_softmax(age_predictions)
	age_probability = np.argmax(age_predictions)
	gender_predictions = np.asarray((gen_model(np.float32(img)))[0])
	gender_predictions = PTA.output_softmax(gender_predictions)
	gender_probability = np.argmax(gender_predictions)

	return age_list[age_probability],age_predictions[0][age_probability],gender_list[gender_probability],gender_predictions[0][gender_probability]

def emotion_recognize(simple_image):
	emotion_list    = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise', 'Neutral']
	emo_model = lib_conn('ER')
	img = PTA.preprocess_input(simple_image, "emo")
	emotion_predictions = np.asarray((emo_model(np.float32(img))))
	emotion_predictions = PTA.output_softmax(emotion_predictions)
	emotion_probability = np.argmax(emotion_predictions)
	emotion_probability_normalizaded = PTA.output_normalization(emotion_probability)

	return emotion_list[emotion_probability_normalizaded], emotion_predictions[0][emotion_probability]
