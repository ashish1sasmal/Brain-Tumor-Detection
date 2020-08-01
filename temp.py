import cv2
import numpy as np
import imutils

img = cv2.imread("Datasets/tumors/kmean.jpg",0)
ret,thresh = cv2.threshold(img,83,255,cv2.THRESH_BINARY)
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

for (i,c) in enumerate(cnts):
    if cv2.contourArea(c)>1000:
        cv2.drawContours(img, [c], -1, (1,255,11), 1)
print(len(cnts))
cv2.imshow("Image",img)

cv2.waitKey(0)
cv2.destroyAllWindows()
