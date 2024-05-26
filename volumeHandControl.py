import time
import cv2
import numpy as np
import HandTrackingModule as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

#############################
wcam, hcam = 640, 480
############################

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)
pTime = 0

detector = htm.handDetector(detectionCon=0.7)

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))
# volume.GetMute()
# volume.GetMasterVolumeLevel()
volRange = volume.GetVolumeRange()
minVol = volRange[0]
maxVol = volRange[1]
vol = 0
volbar = 0
volper =0

while True:
    success, img = cap.read()
    img = detector.findHands(img)
    lm_lst = detector.findPosition(img, draw=False)
    if len(lm_lst) != 0:
        x1, y1 = lm_lst[4][1], lm_lst[4][2]
        x2, y2 = lm_lst[8][1], lm_lst[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1,y1), 10, (255,0,255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2,y2), (255,0,255), 3)

        length = math.hypot(x2-x1, y2-y1)
        # print(length)
        # Hand range --> 250 - 23
        # Volume range --> -65 - 0

        vol = np.interp(length, [23, 250], [minVol, maxVol])
        volbar = np.interp(length, [23, 250], [400, 150])
        volper = np.interp(length, [23, 250], [0, 100])
        print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < 25:
            cv2.circle(img, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    cv2.rectangle(img, (50,150), (85,400), (255, 0, 0), 3)
    cv2.rectangle(img, (50, int(volbar)), (85, 400), (255, 0, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volper)}%', (40, 450), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(img, f'FPS:{int(fps)}', (40,50), cv2.FONT_ITALIC, 1, (255,0,0), 2)
    cv2.imshow("img", img)
    cv2.waitKey(1)
