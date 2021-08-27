import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands(static_image_mode=False)
mpDraw = mp.solutions.drawing_utils
prevTime = 0
while True:
    success , img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                hImg, wImg, cImg = img.shape
                cx, cy = int(lm.x * wImg), int(lm.y * hImg)
                if id==8:
                    cv2.circle(img, (cx,cy), 20, (0,255,255), cv2.FILLED)
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)
    
    #for Calcuting fps
    currTime = time.time()
    fps= 1/(currTime-prevTime)
    prevTime=currTime
    #To show fps on Camera
    cv2.putText( img,"FPS:"+str(int(fps)), (10,70), cv2.FONT_ITALIC, 3, (0,255,0), 3)      
    cv2.imshow("Image",img)
    cv2.waitKey(1)

cap.release()
cv2.destroyAllWindows()