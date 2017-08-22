import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(True):
    # Capture frame-by-frame
    x = 100
    y = 100
    height = 400
    width = 550
    count = 1
    filename = str(count) + ".png"
    ret, frame = cap.read()
    roi = frame[y:height, x:width]
    
    cv2.imshow('ROI',roi)
    #cv2.rectangle(frame, (100, 100), (550, 400), (0,0,255), 3)
    cv2.imshow('frame',frame)
    cv2.imwrite(filename,frame)
    count+=1
    print filename
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
