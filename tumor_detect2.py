import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from imutils import perspective
import sys

cv2.namedWindow('image', cv2.WINDOW_NORMAL)



def nothing(x):
    pass


def tumor_part(c):
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    if hull_area!=0:
        solidity = float(area)/hull_area
    else:
        solidity=0
    if solidity>0.7 and area>2000:
        return True
    else:
        return False

def blur_image(img):
    blur = cv2.GaussianBlur(img,(5,5),0)
    return blur


def enhance(img):
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    gray = cv2.equalizeHist(img)
    # gray = clahe.apply(blur)
    return gray

def morph(img):
    kernel = np.ones((5,5),np.uint8)
    m = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    return m

def threshold(img,b):
    ret,thresh = cv2.threshold(img,b,255,cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)

    # erosion = cv2.erode(thresh,kernel,iterations = 1)
    # dilation = cv2.dilate(erosion,kernel,iterations = 1)

    return thresh

def contours(img2):
    cnts = cv2.findContours(img2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

def RGB(img):
    return cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)



def process(img3,b):
    gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    blur = blur_image(gray)
    cont = enhance(blur)
    thresh = threshold(cont,b)

    mask = np.ones(img3.shape[:2], dtype="uint8") * 255
    cnts = contours(thresh)
    dilation = RGB(thresh)
    for (i,c) in enumerate(cnts):
        if tumor_part(c):
            # print("so",area,solidity)
            cv2.drawContours(img3, [c], -1, (1,255,11), 2)
            cv2.drawContours(dilation, [c], -1, (0,0,254), 2)
        else:
            cv2.drawContours(mask, [c], -1, 0, -1)


    dilation = cv2.bitwise_and(dilation, dilation, mask=mask)
    return (img3,dilation)



img1 = cv2.imread(f"Datasets/tumors/{sys.argv[1]}",0)
# img, dilation = process(img1,127)
# res =  np.hstack((img, dilation))
# clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
# img1 = clahe.apply(img1)

mask = np.ones(img1.shape[:2], dtype="uint8") * 255



open = morph(img1)
skull = img1-open
clean_skull = skull - open
temp = img1-clean_skull
# cv2.imshow("image",threshold(temp,200))

cnts = contours(skull)
for (i,c) in enumerate(cnts):
    cv2.drawContours(img1, [c], -1, (1,255,11), 2)

print(len(cnts))
# t = cv2.bitwise_and(img1, img1, mask=mask)
cv2.imshow("image",img1)

cv2.waitKey(0)
cv2.destroyAllWindows()
