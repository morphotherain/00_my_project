# blink_detector.py
import dlib
import cv2
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time

class BlinkDetector:
    def __init__(self):
        # EAR置信度
        self.EYE_AR_THRESH = 0.3
        # 低于置信度的连续帧数
        self.EYE_AR_CONSEC_FRAMES = 3
        # 数据帧计数器
        self.COUNTER = 0
        # 眨眼的次数
        self.TOTAL = 0

        # 初始化dlib的人脸检测器，它基于HOG
        self.detector = dlib.get_frontal_face_detector()
        # 创建脸部关键点检测器
        self.predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

        # 得到左右眼的关键点索引，左眼是37~42，右眼是43~48
        (self.lStart, self.lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
        (self.rStart, self.rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

        # 获取摄像头数据
        self.vs = VideoStream(src=0).start()
        time.sleep(1.0)

    def eye_aspect_ratio(self, eye):
        # 计算眼部垂直方向上的2组关键点的欧氏距离
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])

        # 计算眼部水平方向上的1组关键点的欧氏距离
        C = dist.euclidean(eye[0], eye[3])

        # 计算EAR
        ear = (A + B) / (2.0 * C)

        return ear

    def blink_main_loop(self):
        # 获取数据帧，调整大小并进行灰度化
        frame = self.vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 得到脸的位置
        rects = self.detector(gray, 0)

        # 针对每一张脸进行处理
        for rect in rects:
            # 将脸部关键点信息转化成numpy的数组
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # 分别计算左右眼的EAR
            leftEye = shape[self.lStart:self.lEnd]
            rightEye = shape[self.rStart:self.rEnd]
            leftEAR = self.eye_aspect_ratio(leftEye)
            rightEAR = self.eye_aspect_ratio(rightEye)

            # 取平均值
            ear = (leftEAR + rightEAR) / 2.0

            # 分别计算左右眼的凸包
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)

            # 画眼部轮廓
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            if ear < self.EYE_AR_THRESH:
                self.COUNTER += 1
            else:
                if self.COUNTER >= self.EYE_AR_CONSEC_FRAMES:
                    self.TOTAL += 1

                # 重置计数器
                self.COUNTER = 0

            # 显示眨眼的次数和EAR
            cv2.putText(frame, "Blinks: {}".format(self.TOTAL), (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 显示图像
        cv2.imshow("Frame", frame)

        key = cv2.waitKey(1) & 0xFF

        # 按q键，退出循环
        if key == ord("q"):
            self.cleanup()

        return self.TOTAL

    def cleanup(self):
        cv2.destroyAllWindows()
        self.vs.stop()

# 用法示例：
if __name__ == '__main__':
    detector = BlinkDetector()
    while True:
        total_blinks = detector.blink_main_loop()
        print(f"Total Blinks: {total_blinks}")
        time.sleep(0.1)
