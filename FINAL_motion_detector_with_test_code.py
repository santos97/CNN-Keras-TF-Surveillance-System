from imutils.video import VideoStream
import cv2
import tensorflow as tf
import argparse
import datetime
import imutils
import time
import cv2

########################
model = tf.keras.models.load_model("/Users/santoshshet/Downloads/mll/64x3-CNN.model")
CATEGORIES = ["Not human", "Human"] 
def final_test():
	def prepare(test_dir):
		IMG_SIZE = 64  # 50 in txt-based
		img_array = cv2.imread(test_dir, cv2.IMREAD_GRAYSCALE)  # read in the image, convert to grayscale
		new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))  # resize image to match model's expected sizing
		return new_array.reshape(-1, IMG_SIZE, IMG_SIZE, 1)  # return the image with shaping that TF wants.

	
	import glob

	images = glob.glob("/Users/santoshshet/Downloads/mll/test/*.jpg")

	for i in images:
		prediction = model.predict([prepare(i)]) # will be a list in a list.
		print(CATEGORIES[int(prediction[0][0])])
		if(CATEGORIES[int(prediction[0][0])]=="Human"):
			return "Human"
		else:
			return "Car"



########################


#counter = 1
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-a", "--min-area", type=int, default=250, help="minimum area size")
args = vars(ap.parse_args())

if args.get("video", None) is None:
	vs = VideoStream(src=0).start()
	time.sleep(0.5)

else:
	vs = cv2.VideoCapture(args["video"])

firstFrame = None
fgbg = cv2.createBackgroundSubtractorMOG2()

# loop over the frames of the video
while True:
	frame = vs.read()
	frame = frame if args.get("video", None) is None else frame[1]
	text = "Not Detected"
	txt2 = "Car"
	if frame is None:
		break

	frame = imutils.resize(frame, width=700)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	#gray = cv2.GaussianBlur(gray, (21, 21), 0)
	fgmask = fgbg.apply(frame)

	if firstFrame is None:
		firstFrame = gray
		continue

	thresh = cv2.threshold(fgmask, 200, 255, cv2.THRESH_TOZERO)[1]

	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)

	for c in cnts:
		if cv2.contourArea(c) < args["min_area"]:
			continue
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
		roi = frame[y:y+h , x:x+w ]
		if roi.shape[0] <= 70 or roi.shape[1] <= 70 :
			bottom = int(80 - roi.shape[0])  # shape[0] = rows
			right = int(80 - roi.shape[1])  # shape[1] = cols
			top , left = 0 , 0
			value = [0,0,0]
			if top>=0 and bottom>0 and right>0 and left>=0:
				dst = cv2.copyMakeBorder(roi, top, bottom, left, right, cv2.BORDER_CONSTANT, None, value)
				cv2.imwrite("/Users/santoshshet/Downloads/mll/test/img.jpg", dst)
			else :
				continue
		else:
			cv2.imwrite("/Users/santoshshet/Downloads/mll/test/img.jpg" , roi)	
		
		#writing images ..................................................
		cv2.imwrite("/Users/santoshshet/Downloads/mll/test/img.jpg", roi)
		text = "Detected"
		txt2 = final_test()
	# draw the text and timestamp on the frame
	cv2.putText(frame, "Motion : {}".format(text), (10, 20),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, "Prediction : {}".format(txt2), (10, 50),
		cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
	cv2.putText(frame, datetime.datetime.now().strftime("%A %d %B %Y %I:%M:%S%p"), (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)

	cv2.imshow("Security Feed", frame)

	cv2.imshow("Thresh", thresh)
	cv2.imshow('Foreground',fgmask)
	key = cv2.waitKey(1) & 0xFF

	if key == ord("q"):
		break

# cleanup work done here 
vs.stop() if args.get("video", None) is None else vs.release()
cv2.destroyAllWindows()

