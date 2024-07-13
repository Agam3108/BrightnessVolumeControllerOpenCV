import cv2 as cv
import mediapipe as mp
import screen_brightness_control as sbc
from math import hypot
import numpy as np
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Initialize MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(
    static_image_mode=False,
    model_complexity=1,
    min_detection_confidence=0.75,
    min_tracking_confidence=0.75,
    max_num_hands=2
)
Draw = mp.solutions.drawing_utils

# Initialize volume control
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
        #Flip Image
    frame = cv.flip(frame, 1)
    #BGR to RGB Conversion
    frameRGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    Process = hands.process(frameRGB)
    landmarkList = []
    hand_type = None
# if hands are present in image
    if Process.multi_hand_landmarks:
        #detect handmarks
        for idx, handlm in enumerate(Process.multi_hand_landmarks):
            handness = Process.multi_handedness[idx].classification[0].label
            for _id, landmarks in enumerate(handlm.landmark):
                #store height and width of image
                height, width, color_channels = frame.shape
                x, y = int(landmarks.x * width), int(landmarks.y * height)
                landmarkList.append([_id, x, y, handness])
            Draw.draw_landmarks(frame, handlm, mpHands.HAND_CONNECTIONS)
    
    if landmarkList:
        right_hand = [lm for lm in landmarkList if lm[3] == 'Right']
        left_hand = [lm for lm in landmarkList if lm[3] == 'Left']

        if right_hand:
               #store coordinates of thumb
            thumb_tip = next(lm for lm in right_hand if lm[0] == 4)
             #store coordinates of index-finger
            index_tip = next(lm for lm in right_hand if lm[0] == 8)
            x_1, y_1 = thumb_tip[1], thumb_tip[2]
            x_2, y_2 = index_tip[1], index_tip[2]
            
             #draw circle on thumb and index finger
            cv.circle(frame, (x_1, y_1), 7, (0, 255, 0), cv.FILLED)
            cv.circle(frame, (x_2, y_2), 7, (0, 255, 0), cv.FILLED)
             #draw line from tip of thumb finger to index finger
            cv.line(frame, (x_1, y_1), (x_2, y_2), (0, 255, 0), 3)

            L = hypot(x_2 - x_1, y_2 - y_1)
            b_level = np.interp(L, [15, 220], [0, 100])
            sbc.set_brightness(int(b_level))

        if left_hand:
            thumb_tip = next(lm for lm in left_hand if lm[0] == 4)
            index_tip = next(lm for lm in left_hand if lm[0] == 8)
            x_1, y_1 = thumb_tip[1], thumb_tip[2]
            x_2, y_2 = index_tip[1], index_tip[2]

            cv.circle(frame, (x_1, y_1), 7, (0, 0, 255), cv.FILLED)
            cv.circle(frame, (x_2, y_2), 7, (0, 0, 255), cv.FILLED)
            cv.line(frame, (x_1, y_1), (x_2, y_2), (0, 0, 255), 3)

            L = hypot(x_2 - x_1, y_2 - y_1)
            vol_level = np.interp(L, [15, 220], [volume.GetVolumeRange()[0], volume.GetVolumeRange()[1]])
            volume.SetMasterVolumeLevel(vol_level, None)
    
    cv.imshow('Image', frame)
    if cv.waitKey(1) & 0xff == ord('q'):
        break

