import cv2
import mediapipe as mp
import time

capture = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

fingerBases = [1,5,9,13,17]
fingerTips = [4,8,12,16,20]
handbase = 0
previousTime = 0
currentTime = 0

while True:
    success, img = capture.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLandMarks in results.multi_hand_landmarks:
            for id, landmark in enumerate(handLandMarks.landmark):
                # print(id,landmark)
                height, width, channels = img.shape
                centre_X, centre_Y = int(landmark.x*width), int(landmark.y*height)
                # print(centre_X, " " ,centre_Y)
                for tip in fingerTips:
                    if id==tip:
                        cv2.circle(img,(centre_X,centre_Y),7,(255,0,0),cv2.FILLED)
                for base in fingerBases:
                    if id==base:
                        cv2.circle(img,(centre_X,centre_Y),7,(0,255,0),cv2.FILLED)
                if id == 0:
                    cv2.circle(img,(centre_X,centre_Y),12,(0,255,255),cv2.FILLED)

                # useful to track a specific part of a specific finger.
                # cv2.circle(img,(centre_X,centre_Y),size,(color),cv2.FILLED)
            mpDraw.draw_landmarks(img, handLandMarks,mpHands.HAND_CONNECTIONS)


    currentTime = time.time()
    fps = 1/(currentTime-previousTime)
    previousTime = currentTime

    cv2.putText(img,str(int(fps)),(10,70),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,255),2)
    cv2.imshow("YouCam", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
