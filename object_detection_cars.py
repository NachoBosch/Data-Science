import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvlib as cv
from cvlib.object_detection import draw_bbox

image = cv2.imread('C:\Code\Machine_Learning\Data_Science_projects\cars.jpg')
box, label, count = cv.detect_common_objects(image)
print(set(label))
for l,c in zip(label,count):
    print(l,c)
# output = draw_bbox(image, box, label, count)
# plt.imshow(output)
# plt.show()
# print("Number of cars in this image are " +str(label.count('car')))