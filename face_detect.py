import cv2

vid = cv2.VideoCapture("walking.avi")

face= cv2.CascadeClassifier('haarcascade_fullbody.xml')

while(True):
   
    # Capture the video frame by frame
    ret, frame = vid.read()

    img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    anything = face.detectMultiScale(img,1.1,5)

    for (x,y,w,h) in anything:
       cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
             
    cv2.imshow('img',frame)
    key = cv2.waitKey(1)
    if key == 32:
        break

vid.release()

# Destroy all the windows
cv2.destroyAllWindows()

