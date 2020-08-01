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
    print(solidity,area)
    if solidity>0.5 and area>300:
        # print(area)
        return True
    else:
        return False

def blur_image(img):
    # blur = cv2.GaussianBlur(img,(5,5),0)
    blur=cv2.bilateralFilter(img,9,75,75)
    return blur


def enhance(img):
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    gray = cv2.equalizeHist(img)
    # gray = clahe.apply(blur)
    return gray

def threshold(img,b):
    ret,thresh = cv2.threshold(img,b,255,cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
    kernel = np.ones((5,5),np.uint8)
    # erosion = cv2.erode(thresh,kernel,iterations = 1)
    # dilation = cv2.dilate(erosion,kernel,iterations = 1)
    dilation = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    dilation = cv2.dilate(dilation,kernel,iterations = 1)

    return dilation

def contours(img2):
    cnts = cv2.findContours(img2.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    return cnts

def RGB(img):
    return cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)

def k_means(img):
    Z = img.reshape((-1,1))
    Z = np.float32(Z)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 8
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))
    return res2


def process(img3,b):
    gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    blur = blur_image(gray)
    # cont = enhance(blur)
    # cont = enhance(cont)

    # thresh = threshold(cont,b)
    thresh = k_means(blur)
    cv2.imwrite("Datasets/tumors/kmean.jpg",thresh)
    mask = np.ones(img3.shape[:2], dtype="uint8") * 255
    cnts = contours(thresh)
    print(len(cnts))
    dilation = RGB(thresh)
    for (i,c) in enumerate(cnts):
        tumor_part(c)
        cv2.drawContours(img3, [c], -1, (1,255,11), 2)
        cv2.drawContours(dilation, [c], -1, (0,0,254), 2)

        # if tumor_part(c):
        #     # print("so",area,solidity)
        #     cv2.drawContours(img3, [c], -1, (1,255,11), 2)
        #     cv2.drawContours(dilation, [c], -1, (0,0,254), 2)
        # else:
        #     cv2.drawContours(mask, [c], -1, 0, -1)


    dilation = cv2.bitwise_and(dilation, dilation, mask=mask)
    return (img3,dilation)



img1 = cv2.imread(f"Datasets/tumors/{sys.argv[1]}")
org = img1.copy()
img, dilation = process(img1,220)
res =  np.hstack((org,img, dilation))
cv2.imshow("image",res)

# cv2.createTrackbar('Intensity','image',220,240,nothing)

# while True:
#     b =  cv2.getTrackbarPos('Intensity','image')
#     img, dilation = process(img1,b)
#     res =  np.hstack((org,img, dilation))
#
#     cv2.imwrite("Results/result2.jpg",res)
#     cv2.imshow("image",res)
#     img1 = cv2.imread(f"Datasets/tumors/{sys.argv[1]}")
#     org = img1.copy()
#     k = cv2.waitKey(1) & 0xFF
#     if k == ord('q'):
#         break

# plt.subplot(121),plt.imshow(res)
# plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(out)
# plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
# plt.suptitle(methods[0])
# plt.show()

# sol.sort(reverse=True)


# for i in range(len(sol)):
#     print(*sol[i], sep=" ")



#
# plot_image = np.concatenate((img, dilation), axis=1)
# plt.imshow(cv2.cvtColor(plot_image, cv2.COLOR_BGR2RGB))
# plt.show()


# cv2.imshow("Br",img)
#
cv2.waitKey(0)
cv2.destroyAllWindows()
