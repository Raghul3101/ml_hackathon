import cv2
import cvlib as cv
from cvlib.object_detection import draw_bbox
import numpy as np

'''def detect_and_draw_box(filename, model="yolov3-tiny", confidence=0.5):

    img_filepath = f'images/{filename}'

    img = cv2.imread(img_filepath)

    bbox, label, conf = cv.detect_common_objects(img, confidence=0.5, model='yolov3-tiny')

    print(f"========================\nImage processed: {filename}\n")

    for l, c in zip(label, conf):
        print(f"Detected object: {l} with confidence level of {c}\n")

    output_image = draw_bbox(img, bbox, label, conf)

    cv2.imshow(f'images_with_boxes/{filename}', output_image)

    display(Image(f'images_with_boxes/{filename}'))

imag = r"C:\Users\LENOVO\OneDrive\Pictures\Wallpapers and PFP\apple.jpg"
detect_and_draw_box(imag)








gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
labels = []
bbox,label,conf = cv.detect_common_objects(img)
print(bbox,label,conf)
out = draw_bbox(img,bbox,label,conf)
cv2.imshow("Out",out)
for item in label:
    if item in labels:
        pass
    else:
        labels.append(item)
cv2.imwrite(f'images_with_boxes/{filename}', output_image)
display(Image(f'images_with_boxes/{filename}'))
cv2.waitKey(0)
print(labels)'''


edges  = cv2.Canny(gray,50,150,apertureSize=3)
cv2.imshow("gojo",edges)
cv2.waitKey(0)'''
'''gray_blurred = cv2.blur(gray, (3, 3))
detected_circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1, 20, param1 = 50,param2 = 30, minRadius = 1, maxRadius = 40)
if detected_circles is not None:
    detected_circles = np.uint16(np.around(detected_circles))
    for pt in detected_circles[0, :]:
        a, b, r = pt[0], pt[1], pt[2]
        cv2.circle(img, (a, b), r, (0, 0, 0), 2)
        cv2.circle(img, (a, b), 1, (0, 0, 0), 3)
        cv2.imshow("Detected Circle", img)
        cv2.waitKey(0)