# Python code for Multiple Color Detection
import numpy as np
import time
import threading


import cv2
#from imutils.video import VideoStream
from directkeys import W, A, S, D
from directkeys import PressKey, ReleaseKey

time.sleep(10)
green_lower = np.array([52, 114, 51], np.uint8)
green_upper = np.array([102, 253, 253], np.uint8)
kernal = np.ones((5, 5), "uint8")



def hold_W(key,hold_time):
    start = time.time()
    PressKey(key)
    while True:
        
        if  time.time() - start > hold_time :
            ReleaseKey(key)
            break



def hold_D_A(key,hold_time):
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

def straight():
    hold_W(W,2)




def left():
    hold_D_A(A, 2)



def right():
    hold_D_A(D, 2)








# Capturing video through webcam
webcam = cv2.VideoCapture(0 , cv2.CAP_DSHOW)





def car():
    while (1):
        _, imageFrame = webcam.read()
       
        hsvFrame = cv2.cvtColor(imageFrame, cv2.COLOR_BGR2HSV)
   
        green_mask = cv2.inRange(hsvFrame, green_lower, green_upper)

    
        green_mask = cv2.dilate(green_mask, kernal)
        res_green = cv2.bitwise_and(imageFrame, imageFrame,
                                mask=green_mask)


    

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





if __name__ == "__main__":


	# creating thread
	t1 = threading.Thread(target=car)
	# t2 = threading.Thread(target=screen, args=(imageFrame))

	# starting thread 1
	t1.start()
    # t2.start()

    


    





   