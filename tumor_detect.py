import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from imutils import perspective
import sys

cv2.namedWindow('image', cv2.WINDOW_NORMAL)



def nothing(x):
    pass

def tumor_part(cnt,area,solidity):
    if solidity>0.7 and area>2000:
        return True
    else:
        return False

def process(img3,b):
    gray = cv2.cvtColor(img3, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(5,5),0)
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8,8))
    gray = clahe.apply(blur)
    # gray = clahe.apply(gray)
    ret,thresh = cv2.threshold(gray,b,255,cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,2)
    kernel = np.ones((10,10),np.uint8)
    erosion = cv2.erode(thresh,kernel,iterations = 1)
    dilation = cv2.dilate(erosion,kernel,iterations = 1)
    #
    # opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    # edges = cv2.Laplacian(dilation,cv2.CV_64F)

    cnts = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)


    sol = []
    contours=[]
    #
    # img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    dilation = cv2.cvtColor(dilation,cv2.COLOR_GRAY2BGR)
    mask = np.ones(img3.shape[:2], dtype="uint8") * 255

    for (i,c) in enumerate(cnts):
        area = cv2.contourArea(c)
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        solidity = float(area)/hull_area
        sol.append([solidity,area])
        if tumor_part(c, area, solidity):
            contours.append(c)
            # print("so",area,solidity)
            cv2.drawContours(img3, [c], -1, (1,255,11), 2)
            cv2.drawContours(dilation, [c], -1, (0,0,254), 2)
        else:
            cv2.drawContours(mask, [c], -1, 0, -1)


    dilation = cv2.bitwise_and(dilation, dilation, mask=mask)
    return (img3,dilation)



img1 = cv2.imread(f"Datasets/tumors/{sys.argv[1]}")
img, dilation = process(img1,127)
res =  np.hstack((img, dilation))
cv2.imshow("image",res)

cv2.createTrackbar('Intensity','image',0,256,nothing)

while True:
    b =  cv2.getTrackbarPos('Intensity','image')
    img, dilation = process(img1,b)
    res =  np.hstack((img, dilation))


    cv2.imwrite("Results/result2.jpg",dilation)
    cv2.imshow("image",res)
    img1 = cv2.imread(f"Datasets/tumors/{sys.argv[1]}")

    k = cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break

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
# cv2.waitKey(0)
cv2.destroyAllWindows()
