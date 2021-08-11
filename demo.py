import time


#from imutils.video import VideoStream
from directkeys import W, A, S, D
from directkeys import PressKey, ReleaseKey




import pyautogui as p



def hold(key,hold_time):
    start = time.time()
    # p.keyDown(key)
    PressKey(W)
    PressKey(key)
    while True:
        
        if  time.time() - start > hold_time :
            # p.keyUp(key)
            ReleaseKey(key)
            ReleaseKey(W)
            break


hold(D, 50)

