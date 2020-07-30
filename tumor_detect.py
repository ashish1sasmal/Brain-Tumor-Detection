import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from imutils import perspective

def tumor_part(cnt,area,solidity):
    if solidity>0.4 and area>2000:
        return True
    else:
        return False

img = cv2.imread("Datasets/tumors/tumor6.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray = clahe.apply(gray)
blur = cv2.GaussianBlur(gray,(5,5),0)

ret,thresh = cv2.threshold(gray,145,255,cv2.THRESH_BINARY)
kernel = np.ones((6,6),np.uint8)
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
mask = np.ones(img.shape[:2], dtype="uint8") * 255

for (i,c) in enumerate(cnts):
    area = cv2.contourArea(c)
    hull = cv2.convexHull(c)
    hull_area = cv2.contourArea(hull)
    solidity = float(area)/hull_area
    sol.append([solidity,area])
    if tumor_part(c, area, solidity):
        contours.append(c)
        print("so",area,solidity)
        cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
        cv2.drawContours(dilation, [c], -1, (0, 255, 0), 2)
    else:
        cv2.drawContours(mask, [c], -1, 0, -1)


dilation = cv2.bitwise_and(dilation, dilation, mask=mask)

plt.subplot(121),plt.imshow(img)
plt.title('Original')
plt.subplot(122),plt.imshow(dilation)
plt.title('Tumor Brain')
plt.suptitle("Brain Tumor Detection Using Image Processing")
plt.show()

cv2.imwrite("Results/result1.jpg",img)

# plt.subplot(121),plt.imshow(res)
# plt.title('Matching Result'), plt.xticks([]), plt.yticks([])
# plt.subplot(122),plt.imshow(out)
# plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
# plt.suptitle(methods[0])
# plt.show()

sol.sort(reverse=True)


# for i in range(len(sol)):
#     print(*sol[i], sep=" ")



#
# plot_image = np.concatenate((img, dilation), axis=1)
# plt.imshow(cv2.cvtColor(plot_image, cv2.COLOR_BGR2RGB))
# plt.show()


# cv2.imshow("Br",img)
#
# cv2.waitKey(0)
# cv2.destroyAllWindows()
