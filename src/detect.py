from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import os
import argparse
import imutils
import time
import dlib
import cv2
from deepface import DeepFace
import pandas as pd

# compute the Eye Aspect Ratio (ear),
# which is a relation of the average vertical distance between eye landmarks to the horizontal distance
def eye_aspect_ratio(eye):
    vertical_dist = dist.euclidean(eye[1], eye[5]) + dist.euclidean(eye[2], eye[4])
    horizontal_dist = dist.euclidean(eye[0], eye[3])
    ear = vertical_dist / (2.0 * horizontal_dist)
    return ear


BLINK_THRESHOLD = 0.19  # the threshold of the ear below which we assume that the eye is closed
CONSEC_FRAMES_NUMBER = 2  # minimal number of consecutive frames with a low enough ear value for a blink to be detected

# get arguments from a command line
ap = argparse.ArgumentParser(description='Eye blink detection')
ap.add_argument("-p", "--shape-predictor", required=True, help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="", help="path to input video file")
args = vars(ap.parse_args())

# initialize dlib's face detector (HOG-based) and facial landmark predictor
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# choose indexes for the left and right eye
(left_s, left_e) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_s, right_e) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# start the video stream or video reading from the file
video_path = args["video"]
if video_path == "":
    vs = VideoStream(src=0).start()
    print("[INFO] starting video stream from built-in webcam...")
    fileStream = False
else:
    vs = FileVideoStream(video_path).start()
    print("[INFO] starting video stream from a file...")
    fileStream = True
time.sleep(1.0)

counter = 0
total = 0
alert = False
start_time = 0
frame = vs.read()
filename = 'img.jpg'

# loop over the frames of video stream:
# grab the frame, resize it, convert it to grayscale
# and detect faces in the grayscale frame
while (not fileStream) or (frame is not None):
    frame = imutils.resize(frame, width=640)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray_frame, 0)
    ear = 0
    # loop over the face detections:
    # determine the facial landmarks,
    # convert the facial landmark (x, y)-coordinates to a numpy array,
    # then extract the left and right eye coordinates,
    # and use them to compute the average eye aspect ratio for both eyes
    for rect in rects:
        shape = predictor(gray_frame, rect)
        shape = face_utils.shape_to_np(shape)
        leftEye = shape[left_s:left_e]
        rightEye = shape[right_s:right_e]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        # visualize each of the eyes
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        #cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        #cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        # if the eye aspect ratio is below the threshold, increment counter
        # if the eyes are closed longer than for 2 secs, raise an alert
        if ear < BLINK_THRESHOLD:
            counter += 1
            if start_time == 0:
                start_time = time.time()
            else:
                end_time = time.time()
                if end_time - start_time > 2: alert = True
        else:
            if counter >= CONSEC_FRAMES_NUMBER:
                total += 1
            counter = 0
            start_time = 0
            alert = False

    # draw the total number of blinks and EAR value
    #cv2.putText(frame, "Blinks: {}".format(total), (10, 30),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    #cv2.putText(frame, "EAR: {:.2f}".format(ear), (500, 30),
                #cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    if alert:
        cv2.putText(frame, "ALERT!", (150, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # show the frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if total>=1:
        cv2.imwrite(filename, frame)

        df = DeepFace.find(img_path = "img.jpg", db_path = "file/my_db")
        #print(df)

        if df.empty:
            print('no match')

        else:
            string=df['identity'][0]

            path=os.path.dirname(string)
            name=os.path.basename(path)
            print('Identified as :', name)

        
        time.sleep(2)
        break 
    
    # if the `q` key was pressed, break from the loop
    if key == ord("q"):
        break
    frame = vs.read()

cv2.destroyAllWindows()
vs.stop()
