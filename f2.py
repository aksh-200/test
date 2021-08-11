# Python code for Multiple Color Detection
import numpy as np
import time

import cv2
#from imutils.video import VideoStream
from directkeys import W, A, S, D
from directkeys import PressKey, ReleaseKey

time.sleep(10)
green_lower = np.array([52, 114, 51], np.uint8)
green_upper = np.array([102, 253, 253], np.uint8)
kernal = np.ones((5, 5), "uint8")
# Capturing video through webcam
webcam = cv2.VideoCapture(0 , cv2.CAP_DSHOW)


def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)


def left():
    PressKey(W)

    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)
   




def right():
    PressKey(W)

    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(S)
    



def stop():
    PressKey(S)
    ReleaseKey(W)
    ReleaseKey(A)
    ReleaseKey(D)






# Start a while loop
while (1):


 #  start = time.time()
    # Reading the video from thepy
    # webcam in image frames
    _, imageFrame = webcam.read()


    #imageFrame = cv2.flip(imageFrame,1)

    # Convert the imageFrame in s
    # BGR(RGB color space) to
    # HSV(hue-saturation-value)
    # color space
    hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
    # Set range for green color and
    # define mask
    #green_lower = np.array([52, 114, 51], np.uint8)
    #green_upper = np.array([102, 253, 253], np.uint8)
    green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color
    #kernal = np.ones((5, 5), "uint8")

    # For green color
    green_mask = cv2.dilate(green_mask, kernal)
    res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                mask=green_mask)


    # Creating contour to track green color

    contours, hierarchy = cv2.findContours(green_mask.copy(),
                                           cv2.RETR_TREE,
                                           cv2.CHAIN_APPROX_SIMPLE)

    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

    try:

        m1 = cv2.moments(sorted_contours[0])
        m2 = cv2.moments(sorted_contours[1])
        x1 = int(m1["m10"] / m1["m00"])
        y1 = int(m1["m01"]  / m1["m00"])
        x2 = int(m2["m10"] / m2["m00"])
        y2 = int(m2["m01"] / m2["m00"])
        print("x1 {} ,x2 {} , y1 {} ,y2 {}".format(x1,x2,y1,y2))
        print(x1+x2)
        slope = (y2 - y1) / (x2 - x1)
        cv2.line(imageFrame, (x1, y1), (x2, y2), (87, 200, 10), 4)
        PressKey(W)
        if -0.50 < slope < 0:

            print("slope is negative {}".format(slope))
            straight()



            '''PressKey(RIGHT)
            ReleaseKey(UP)
            ReleaseKey(RIGHT)'''



            #cv2.putText(imageFrame, "negatve", (int((x1 + x2) / 2), int((y1 + y2) / 2)),
                        #cv2.FONT_HERSHEY_SIMPLEX,
                        #1.0, (0, 0, 255))


        elif slope < -0.50 :

            print("slope is negative {}".format(slope))
            #PressKey(W)

            right()







        elif 0 < slope <0.50:
            print("slope is positive{}".format(slope))
            straight()
            #cv2.putText(imageFrame, "postve", (int((x1 + x2) / 2), int((y1 + y2) / 2)),
             #           cv2.FONT_HERSHEY_SIMPLEX,
              #          1.0, (0, 0, 255))

        else:

            print("slope is positive{}".format(slope))
            left()





    except IndexError:
        pass

    except ZeroDivisionError:

        stop()
       
    # Program Termination

    cv2.imshow("Multiple Color Detection in Real-TIme", imageFrame)
#    end = time.time()
 #   print("total tme {}".format(end - start))

    if cv2.waitKey(1) & 0xFF == ord('q'):

        cap.release()
        cv2.destroyAllWindows()
        break






