import cv2
import numpy as np
net = cv2.dnn.readNetFromTensorflow("frozen_inferene_graph_coco.pb","mask_rcnn_inception_v2_coco_2018_01_28.pbtxt")
img = cv2.imread("person.jpg")
blob = cv2.dnn.blobFromImage(img, swapRB=True)
net.setInput(blob)
boxes, masks = net.forward(["detection_out_final", "detection_masks"])
print(boxes)
'''detection_count = boxes.shape[2]
for i in range(detection_count):
    box = boxes[0, 0, i]
    class_id = box[1]
    score = box[2]
    if score < 0.5:
        continue
    x = int(box[3] * width)
    y = int(box[4] * height)
    x2 = int(box[5] * width)
    y2 = int(box[6] * height)
    roi = black_image[y: y2, x: x2]
    roi_height, roi_width, _ = roi.shape'''