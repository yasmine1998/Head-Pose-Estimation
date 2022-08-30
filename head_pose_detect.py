from imutils.video import VideoStream
from face_detector import find_faces, get_face_detector, get_onnx_face_detector, find_faces_onnx
from eye_tracker import contouring, eye_on_mask, print_eye_pos, process_thresh
from face_landmarks import get_square_box,get_landmark_model,detect_marks,get_dlib_landmark_model,detect_marks_dlib
from imutils import face_utils
from head import *
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2


#detector = dlib.get_frontal_face_detector()
onnx_model = get_onnx_face_detector()
#landmark_model = get_landmark_model()
landmark_model = get_dlib_landmark_model()
vs = VideoStream(src=0).start()

font = cv2.FONT_HERSHEY_SIMPLEX
left = [2,3]
right = [0,1]
kernel = np.ones((9, 9), np.uint8)
model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            (0.0, -330.0, -65.0),        # Chin
                            (-225.0, 170.0, -135.0),     # Left eye left corner
                            (225.0, 170.0, -135.0),      # Right eye right corner
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner
                        ])

time.sleep(2.0)

while True:
	# grab the frame from the threaded video stream, resize it to
	# have a maximum width of 400 pixels, and convert it to
	# grayscale
        frame = vs.read()
	#frame = imutils.resize(frame, width=400)
	#gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
	# detect faces in the grayscale frame
	#rects = detector(gray, 0)
        size = frame.shape
        focal_length = size[1]
        center = (size[1]/2, size[0]/2)
        camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
        rects = find_faces_onnx(frame, onnx_model)
	# check to see if a face was detected, and if so, draw the total
	# number of faces on the frame
        if not rects:
	        cv2.putText(frame, "Away from the camera", (10, 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
        if len(rects) > 0:
	        text = "{} face(s) found".format(len(rects))
	        cv2.putText(frame, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX,0.5, (0, 0, 255), 2)
		
        for rect in rects:
	        marks = detect_marks_dlib(frame,landmark_model,rect)
	       
	        mark_chin = np.array([ marks[4][0],marks[4][1]-(marks[2][1]-marks[4][1]) ], dtype="double")
	        mark_mouth_l = np.array([ mark_chin[0]-(mark_chin[0]-marks[3][0]),marks[4][1]-((marks[4][1]-mark_chin[1])/2) ], dtype="double")
	        mark_mouth_r = np.array([ mark_chin[0]+(marks[1][0]-mark_chin[0]),marks[4][1]-((marks[4][1]-mark_chin[1])/2) ], dtype="double")
	        image_points = np.array([marks[4],mark_chin, marks[2],marks[0],mark_mouth_l,mark_mouth_r ], dtype="double")
	        
	      
	        dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
	        rvec = np.zeros(3, dtype=np.float)
	        tvec = np.array([0, 0, 1], dtype=np.float)
	        
	        (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_UPNP)
	        (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
	        for p in image_points:
	                cv2.circle(frame, (int(p[0]), int(p[1])), 3, (0,0,255), -1)
	                p1 = ( int(image_points[0][0]), int(image_points[0][1]))
	                p2 = ( int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
	                x1, x2 = head_pose_points(frame, rotation_vector, translation_vector, camera_matrix)
	                cv2.line(frame, p1, p2, (0, 255, 255), 2)
	                cv2.line(frame, tuple(x1), tuple(x2), (255, 255, 0), 2)
	                try:
	                    m = (p2[1] - p1[1])/(p2[0] - p1[0])
	                    ang1 = int(math.degrees(math.atan(m)))
	                except:
	                    ang1 = 90
	                    
	                try:
	                    m = (x2[1] - x1[1])/(x2[0] - x1[0])
	                    ang2 = int(math.degrees(math.atan(-1/m)))
	                except:
	                    ang2 = 90
	                    
	                if ang2 >= 40:
	                    print('Head right')
	                    msg = 'Head right'
	                    cv2.putText(frame, 'Head right', (90, 30), font, 2, (255, 255, 128), 3)
	                elif ang2 <= -35:
	                    print('Head left')
	                    msg = 'Head left'
	                    cv2.putText(frame, 'Head left', (90, 30), font, 2, (255, 255, 128), 3)
	                cv2.putText(frame, str(ang1), tuple(p1), font, 2, (128, 255, 255), 3)
	                cv2.putText(frame, str(ang2), tuple(x1), font, 2, (255, 255, 128), 3)
	                
	        
		# loop over the (x, y)-coordinates for the facial landmarks
		# and draw each of them
	        #for (i, (x, y)) in enumerate(shape):
		        #cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
		        #cv2.putText(frame, str(i + 1), (x - 10, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 255), 1)
			
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
 
	# if the `q` key was pressed, break from the loop
        if key == ord("q"):
	        break
		
cv2.destroyAllWindows()
vs.stop()
