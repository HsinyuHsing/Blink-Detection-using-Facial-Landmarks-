from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np
import argparse
import time
import dlib # 人臉檢測器和 68 點人臉標誌檢測器
import cv2

FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

# http://vision.fe.uni-lj.si/cvww2016/proceedings/papers/05.pdf
def eye_aspect_ratio(eye):
	# 計算眼睛距離
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])
	# 计算距离，水平的
	C = dist.euclidean(eye[0], eye[3])
	# ear值
	ear = (A + B) / (2.0 * C)
	return ear
 # EAR= (垂直距離 A+垂直距離 B)/2×水平距離 C
 # 眼睛睜開時 EAR 值較大，閉眼時 EAR 值降低，因此可以透過 EAR 變化判斷是否閉眼

# 輸入參數
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
	help="path to facial landmark predictor")
ap.add_argument("-v", "--video", type=str, default="",
	help="path to input video file")
args = vars(ap.parse_args())
 


# 設定判斷參數
# EYE_AR_THRESH = 0.3：當 EAR 低於 0.3 時視為閉眼。
# EYE_AR_CONSEC_FRAMES = 3：需要 連續 3 幀 閉眼才算真正眨眼。
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3


# 初始化計數
COUNTER = 0
TOTAL = 0

# 偵測和定位工具
print("[INFO] loading facial landmark predictor...")
# detector：使用 dlib 提供的 HOG+SVM 臉部偵測器來偵測人臉。
# predictor：使用 形狀預測模型 來標記臉部的 68 個關鍵點。
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])

# 分别取兩個眼睛區域
(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

# 讀取影片
print("[INFO] starting video stream thread...")
vs = cv2.VideoCapture(args["video"])
#vs = FileVideoStream(args["video"]).start()
time.sleep(1.0)

def shape_to_np(shape, dtype="int"):
	# 創建68*2
	coords = np.zeros((shape.num_parts, 2), dtype=dtype)
	# 遍歷每一个關鍵點
	# 得到坐標
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)
	return coords

# 遍歷每一幀
while True:
	# 預處理
	frame = vs.read()[1]
	if frame is None:
		break
	
	(h, w) = frame.shape[:2]
	width=1200
	r = width / float(w)
	dim = (width, int(h * r))
	frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	# 檢測人臉
	rects = detector(gray, 0)

	# 遍歷每個偵測到的人臉
	for rect in rects:
		# 取得座標
		shape = predictor(gray, rect)
		shape = shape_to_np(shape)

		# 分别計算ear值
		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)

		# 計算平均
		ear = (leftEAR + rightEAR) / 2.0

		# 繪製閉眼區域
		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		# 檢查是否滿足
		if ear < EYE_AR_THRESH:
			COUNTER += 1

		else:
			# 如果連續幾偵都是閉眼的，總數算一次
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1

			# 重置
			COUNTER = 0

		# 顯示
		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(10) & 0xFF
 
	if key == 27:
		break

vs.release()
cv2.destroyAllWindows()


# python detect_blinks.py -p shape_predictor_68_face_landmarks.dat -v test.mp4

# python detect_blinks.py -p shape_predictor_68_face_landmarks.dat -v tom-cruise-and-brad-pitt.mp4

# python detect_blinks.py -p shape_predictor_68_face_landmarks.dat -v shoheiohtani.mp4

