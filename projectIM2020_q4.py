
# coding: utf-8

# In[4]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
img = cv2.imread("Q4Images/5.jpeg")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img2=img.copy()
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)
def Show_img(img,img2):
    vis = np.concatenate((img,img2), axis=1)
    
    cv2.imshow('Image with Borders', vis) 
# De-allocate any associated memory usage  
    if cv2.waitKey(0): 
        cv2.destroyAllWindows()
def Draw_circles(circles):                             
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            if(r!=0):
                cv2.circle(img, (x, y), r, (10, 255, 0), 2)
                cv2.rectangle(img, (x - 1, y - 1), (x + 1, y + 1), (110, 0, 255), -1)          
def find_circles(img): 
    n = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  
    circles=cv2.HoughCircles(n, cv2.HOUGH_GRADIENT, 1.1, minDist=2,param1=400,param2=16,minRadius=1,maxRadius=7)
    Draw_circles(circles)
    temp=adjust_gamma(img,2) 
    n = cv2.cvtColor(temp, cv2.COLOR_BGR2GRAY)
    circles=cv2.HoughCircles(n, cv2.HOUGH_GRADIENT, 1.1, minDist=2,param1=400,param2=16,minRadius=1,maxRadius=7)
    Draw_circles(circles)
           
find_circles(img2)
Show_img(img,img2) 

   

