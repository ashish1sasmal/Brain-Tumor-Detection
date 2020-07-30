import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from imutils import perspective

def tumor_part(cnt,area,solidity):
    if solidity>0.4 and area>800:
        return True

img = cv2.imread("Datasets/tumors/tumor2.jpg",0)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img = clahe.apply(img)
blur = cv2.GaussianBlur(img,(5,5),0)

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)

ret,thresh = cv2.threshold(img,130,255,cv2.THRESH_BINARY)
kernel = np.ones((6,6),np.uint8)
erosion = cv2.erode(thresh,kernel,iterations = 1)
dilation = cv2.dilate(erosion,kernel,iterations = 1)
#
# opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
# edges = cv2.Laplacian(dilation,cv2.CV_64F)

cnts = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
dilation = cv2.cvtColor(dilation,cv2.COLOR_GRAY2BGR)

sol = []

for (i,c) in enumerate(cnts):
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    sol.append([solidity,area])
    if tumor_part(c, area, solidity):
    # if cv2.contourArea(c)>3000 and cv2.contourArea(c)<4000:

        #
        print("So",solidity,cv2.contourArea(c))

        # epsilon = 0.8*cv2.arcLength(c,True)
        # box = cv2.approxPolyDP(c,epsilon,True)

        # box = cv2.minAreaRect(c)
        # box = cv2.cv.BoxPoints(box) if imutils.is_cv2() else cv2.boxPoints(box)
        # box = np.array(box, dtype="int")
        # # print("Object #{}".format(ind))
        # # print(box)
        # # print()
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        cv2.drawContours(dilation, [c], -1, (0, 255, 0), 2)
        #
        # box = perspective.order_points(box)
        # # for (x, y) in rect:
        # # 	cv2.putText(out, "Object #{}".format(ind),(int(rect[0][0] - 15), int(rect[0][1] - 15)),cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 0, 255), 2)
        #
        # (tl, tr, br, bl) = box
        # (tltrX, tltrY) = midpoint(tl, tr)
        # (blbrX, blbrY) = midpoint(bl, br)
        # # compute the midpoint between the top-left and top-right points,
        # # followed by the midpoint between the top-righ and bottom-right
        # (tlblX, tlblY) = midpoint(tl, bl)
        # (trbrX, trbrY) = midpoint(tr, br)
        # # draw the midpoints on the image
        # cv2.circle(img, (int(tltrX), int(tltrY)), 5, (255, 0, 0), -1)
        # cv2.circle(img, (int(blbrX), int(blbrY)), 5, (255, 0, 0), -1)
        # cv2.circle(img, (int(tlblX), int(tlblY)), 5, (255, 0, 0), -1)
        # cv2.circle(img, (int(trbrX), int(trbrY)), 5, (255, 0, 0), -1)



plt.subplot(121),plt.imshow(img)
plt.title('Original')
plt.subplot(122),plt.imshow(dilation)
plt.title('Tumor Brain')
plt.suptitle("Brain Tumor Detection Using Image Processing")
plt.show()

# plt.subplot(121),plt.imshow(res)
# plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(out)
# plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
# plt.suptitle(methods[0])
# plt.show()

sol.sort(reverse=True)


for i in range(len(sol)):
    print(*sol[i], sep=" ")



#
# plot_image = np.concatenate((img, dilation), axis=1)
# plt.imshow(cv2.cvtColor(plot_image, cv2.COLOR_BGR2RGB))
# plt.show()


# cv2.imshow("Br",img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
