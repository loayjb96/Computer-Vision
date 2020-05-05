
# coding: utf-8

# In[272]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
img1 = cv2.imread('Q1DB/2.jpeg')
img2 = cv2.imread('Q1DB/11.jpeg')
def Show_img(vis,title):
    vis = cv2.resize(vis, (1200, 400), None, .12, .15)
    cv2.imshow(title, vis) 
# De-allocate any associated memory usage  
    if cv2.waitKey(0): 
        cv2.destroyAllWindows() 
def concatenate(img,img1,new_img):
    img = cv2.resize(img, (420, 627), None, .12, .15) 
    img1 = cv2.resize(img1, (420, 627), None, .12, .15) 
    new_img = cv2.resize(new_img, (420, 627), None, .12, .15)   
    vis = np.concatenate((new_img, img,img1), axis=1)
    return vis
def find_t(original_image,kx,ky,multy):
    image = original_image.copy()  
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kx,ky))
    dilate = cv2.dilate(thresh, kernel, iterations=kx)
    dilate = cv2.erode(dilate, kernel, iterations=ky)
    dilate=binarize(dilate,250)
# Find contours
    cnts = cv2.findContours(dilate, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    mask = np.zeros(image.shape, dtype='uint8')
    mask.fill(255)
# Iterate thorugh contours and filter for ROI
    for c in cnts:
        area = cv2.contourArea(c)
        if area < multy and area > 3000:
            x,y,w,h = cv2.boundingRect(c)  
            if(h!=66 and h!=65 and h!=62 and h!=73 and h!=71 and h!=76):
                mask[y:y+h, x:x+w] = original_image[y:y+h, x:x+w]
    
    return dilate,mask 
def find_conturs2(img,originalImage):
    temp=img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=binarize(img,250)
    corners = cv2.goodFeaturesToTrack(img,900,0.01,20)
    corners = np.int64(corners)
    j=0
    for i in corners:
        j+=1
        x,y = i.ravel()
        if(j==19 or j==10):
            cv2.circle(originalImage,(x+85,y),10,(0,0,255),-1)
        if(j==21 and x>2000):
            cv2.circle(originalImage,(x+140,y),10,(0,0,255),-1)
            cv2.circle(originalImage,(x-1820,y-95),10,(0,0,255),-1)
        cv2.circle(originalImage,(x,y),10,(0,0,255),-1)
    return originalImage
def binarize(image_to_transform, threshold):
    # now, lets convert that image to a single greyscale image using convert()
    output_image=image_to_transform.copy()
    arr = np.array(output_image)
    arr[arr <=threshold] = 0
    arr[arr >threshold] = 100
    return arr
#####GAMMA correction for dark images for better results
def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
    for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)                                 # change the value here to get different result
def cut_til_edge(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    cnts1,cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    new_img=image
    for c in cnts:
        area = cv2.contourArea(c)
        x,y,w,h = cv2.boundingRect(c)
        if area==4572159.0   :
            new_img=image[y:y+h,x+w//6:x+w]
            return new_img          
        if area==5587046.5:
            new_img=image[y+450:y+h,x+250:x+w]
    return new_img
############## calling the above functions ##################
addjusted = adjust_gamma(img1, gamma=0.87)
addjusted2 = adjust_gamma(img2, gamma=0.87)
gray1=binarize(img1,250)
gray2=binarize(addjusted2,80)
dilate1,mask1=find_t(gray1,7,5,9000)
dilate2,mask2=find_t(gray2,5,3,3850)
############## finding the edges and mark them ##############
img3=find_conturs2(mask1,addjusted)
img4=find_conturs2(mask2,addjusted2)
############## finding the edgeless img #####################
new1=cut_til_edge(img1)
new2=cut_til_edge(img2)
############## concatinating the 3 images ###################
final=concatenate(img3,img1,new1)
final2=concatenate(img4,img2,new2)
Show_img(final,"Output for Question 1 image 1 press any key to show the secound output")
Show_img(final2,"Output for Question 1 image 2")

