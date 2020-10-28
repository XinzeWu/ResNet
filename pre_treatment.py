import sys
import cv2
import os
pic_path = 'E:/university/project/daoshizhi/data/'
save_path = "E:/university/project/daoshizhi/data2/"
for i in range(7):
    pic_name = os.listdir(pic_path + str(i))
    count = 0
    for j in pic_name:
        img = cv2.imread(pic_path + str(i) + "/" + j ,cv2.IMREAD_GRAYSCALE)
        img  = cv2.resize(img,(224,224),interpolation=cv2.INTER_AREA)
        cv2.imwrite(save_path+str(i)+"/" + "{}.png".format(count), img, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        count=count + 1
