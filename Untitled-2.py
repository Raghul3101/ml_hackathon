import cv2
import matplotlib.pyplot as mp
config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = 'frozen_inference_graph.pb'
model = cv2.dnn_DetectionModel(frozen_model,config_file)
with open('labels.txt','rt') as l:
    label_arr = l.read().rstrip('\n').split('\n')
model.setInputSize(360,360)
model.setInputScale(1.0/127.5)
model.setInputMean((127.5,127.5,127.5))
model.setInputSwapRB(True)
img = cv2.imread('knife.jpg')
mp.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

ci,confidence,bbox = model.detect(img,confThreshold = 0.6)
print(ci)
'''for i in range(len(ci)):
    print(label_arr[ci[i]-1])'''
font_scale = 3
font = cv2.FONT_HERSHEY_PLAIN
for boxes in bbox:
    cv2.rectangle(img,boxes,(0,255,0),2)
for class_ind,conf,boxes in zip(ci.flatten(),confidence.flatten(),bbox):
    cv2.rectangle(img,boxes,(0,255,0),2)
mp.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
mp.waitforbuttonpress()