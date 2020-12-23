from cv2 import cv2
from scipy.spatial.distance import euclidean
import imutils
from imutils import perspective
from imutils import contours
import numpy as np
import utlis

scale=1
wP=680*scale
hP=680*scale
path = 'output.jpg'

def geContours(img,imgContour):
    contours, heirarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        areaMin = 100
        if area > areaMin:
            cv2.drawContours(imgContour, cnt, -1, (255, 0, 255), 5)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            x , y , w, h = cv2.boundingRect(approx)
            cv2.rectangle(imgContour, (x , y ), (x + w , y + h ), (0, 255, 0), 5)
            
            M = cv2.moments(cnt)
            cx = int(M["m10"]/M["m00"])
            cy = int(M["m01"]/M["m00"])

            cv2.circle(imgContour, (cx, cy), 7, (255, 255, 255), -1)

            rotrect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rotrect)
            box = np.int0(box)

            angle = rotrect[-1]
            if angle < -45:
                angle = -(90 + angle)
            else:
                angle = -angle

            print((round(angle)),"deg")
            print((cx, cy),"")
 
            cv2.putText(imgContour, "Area: " + str(int(area)), (x + w + 20, y + 35), cv2.FONT_HERSHEY_COMPLEX, 0.4,
                        (0, 255, 0), 1)
            cv2.putText(imgContour, "Degree: " + str(int(angle)), (x + w + 20, y + 50), cv2.FONT_HERSHEY_COMPLEX, 0.4,
                        (0, 255, 0), 1)

while True:
    img = cv2.imread(path)
    imgContours , conts = utlis.getContours(img,minArea=5000,filter=4)
    if len(conts) != 0:
        biggest = conts[0][2]
        #print(biggest)
        imgWarp = utlis.warpImg(img, biggest, wP,hP)
        imgContours2, conts2 = utlis.getContours(imgWarp,
                                                 minArea=100, filter=4,
                                                 cThr=[50,50],draw = False)
        if len(conts) != 0:
            for obj in conts2:
                imgContour = imgContours2.copy()
                imgBlur = cv2.GaussianBlur(imgContours2, (9, 9), 1)
                imgGray = cv2.cvtColor(imgBlur, cv2.COLOR_BGR2GRAY)
                edged = cv2.Canny(imgBlur, 50, 100)
                edged = cv2.dilate(edged, None, iterations=1)
                edged = cv2.erode(edged, None, iterations=1)
                cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = imutils.grab_contours(cnts)
                (cnts, _) = contours.sort_contours(cnts)
                cnts = [x for x in cnts if cv2.contourArea(x) > 100]
                ref_object = cnts[0]
                box = cv2.minAreaRect(ref_object)
                box = cv2.boxPoints(box)
                box = np.array(box, dtype="int")
                box = perspective.order_points(box)
                (tl, tr, br, bl) = box
                dist_in_pixel = euclidean(tl, tr)
                dist_in_cm = 25
                pixel_per_cm = dist_in_pixel/dist_in_cm
                pixel_per_cm = dist_in_pixel/dist_in_cm
                threshold1 = 200
                threshold2 = 2
                imgCanny = cv2.Canny(imgGray,threshold1,threshold2)
                kernel = np.ones((5, 5))
                imgDil = cv2.dilate(imgCanny, kernel, iterations=1)
                geContours(imgDil,imgContour)
                for cnt in cnts:
                    box = cv2.minAreaRect(cnt)
                    box = cv2.boxPoints(box)
                    box = np.array(box, dtype="int")
                    box = perspective.order_points(box)
                    (tl, tr, br, bl) = box
                    mid_pt_horizontal = (tl[0] + int(abs(tr[0] - tl[0])/2), tl[1] + int(abs(tr[1] - tl[1])/2))
                    mid_pt_verticle = (tr[0] + int(abs(tr[0] - br[0])/2), tr[1] + int(abs(tr[1] - br[1])/2))
                    wid = euclidean(tl, tr)/pixel_per_cm
                    ht = euclidean(tr, br)/pixel_per_cm
                    cv2.putText(imgContour, "{:.1f}cm".format(wid), (int(mid_pt_horizontal[0] - 15), int(mid_pt_horizontal[1] - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    cv2.putText(imgContour, "{:.1f}cm".format(ht), (int(mid_pt_verticle[0] + 10), int(mid_pt_verticle[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    print(round((wid)),"cm", round( ht),"cm")
                    cv2.imshow("Result", imgContour)
 
    img = cv2.resize(img,(0,0),None,0.8,0.8)
    cv2.imshow('Original',img)
    k = cv2.waitKey(0)
    if k == 27:
        break         
cv2.destroyAllWindows()