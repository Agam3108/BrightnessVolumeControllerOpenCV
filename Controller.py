import cv2 as cv
import mediapipe as mp
import screen_brightness_control as sbc
from math import hypot
import numpy as np

#Initializing the model
mpHands=mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity =1,
    min_detection_confidence = 0.75,
    min_tracking_confidence = 0.75,
    max_num_hands = 2

)
Draw = mp.solutions.drawing_utils

cap=cv.VideoCapture(0)

while True:
    _,frame = cap.read()
    #Flip Image
    frame= cv.flip(frame,1)
    #Convert BGR to RGB
    frameRGB = cv.cvtColor(frame,cv.COLOR_BGR2RGB)

    Process = hands.process(frameRGB)
    landmarkList =[]
    # if hands are present in image
    if Process.multi_hand_landmarks:
        #detect handmarks
        for handlm in Process.multi_hand_landmarks:
            for _id,landmarks in enumerate(handlm.landmark):
                #store height and width of image
                height,width,color_channels = frame.shape

                x,y =int(landmarks.x*width),int(landmarks.y*width)
                landmarkList.append([_id,x,y])
            Draw.draw_landmarks(frame,handlm,mpHands.HAND_CONNECTIONS)
    if landmarkList !=[]:
        #store coordinates of thumb
        x_1,y_1 = landmarkList[4][1],landmarkList[4][2]
        #store coordinates of index-finger
        x_2,y_2 = landmarkList[8][1],landmarkList[8][2]

        #draw circle on thumb and index finger
        cv.circle(frame,(x_1,y_1),7,(0,255,0),cv.FILLED)
        cv.circle(frame,(x_2,y_2),7,(0,255,0),cv.FILLED)

        #draw line from tip of thumb finger to index finger
        cv.line(frame,(x_1,y_1),(x_2,y_2),(0,255,0),3)

        L=hypot(x_2-x_1,y_2-y_1)

        b_level = np.interp(L,[15,220],[0,100])

        sbc.set_brightness(int(b_level))
    cv.imshow('Image',frame)
    if cv.waitKey(1) & 0xff == ord('q'):
        break

        
            
          
     

