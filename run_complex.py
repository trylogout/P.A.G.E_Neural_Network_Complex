from utils.cuda import check
check()
import cv2
import argparse
import numpy as np
import utils.util_loader as UL

def print_res(a_prob,
			a_pred,
			g_prob,
			g_pred, 
			em_prob, 
			em_pred, 
			personality):
	print("\nFull result about person:")
	print("Age: {0}, probability: {1:.2f}%".format(a_prob,a_pred))
	print("Gender: {0}, probability: {1:.2f}%".format(g_prob,g_pred))
	print("Emotion: {0}, probability: {1:.2f}%".format(em_prob, em_pred))
	print('Same person probability: {0:.2f}%\n'.format(personality))

def main(person_image, input_image):
	person_image = cv2.imread(person_image)
	input_image = cv2.imread(input_image)
	a_prob,a_pred,g_prob,g_pred = UL.age_gender_reconize(input_image)
	em_prob, em_pred = UL.emotion_recognize(input_image)
	personality = UL.person_recognize(person_image, input_image)
	print_res(a_prob,a_pred,g_prob,g_pred, em_prob, em_pred, personality)

if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("-pm", 
						"--person_image", 
						type = str, 
						help = "Image to emulate data from DB.", 
						required = True)
	parser.add_argument("-im", 
						"--input_image", 
						type = str, 
						help = "Input image with face.", 
						required = True)
	args = parser.parse_args()
	main(args.person_image, args.input_image)