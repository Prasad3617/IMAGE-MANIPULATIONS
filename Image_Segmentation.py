#!/usr/bin/env python
# coding: utf-8

# # Image_Segmentation

# ## 1. Contours

# ### Contours are continuous lines or curves that bound or cover the full-boundary of an object in an image

# In[1]:


import cv2                                       #importing necessary libraries
import numpy as np


# In[2]:


image=cv2.imread("2.jpg")                        #loading our image


# In[3]:


cv2.imshow("Input Image", image)                 #displaying our original image
cv2.waitKey(0)


# In[4]:


gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)      #displaying the grayscale-image


# In[5]:


edged=cv2.Canny(gray,30,200)                     #finding canny edges
cv2.imshow("Canny Edges",edged)
cv2.waitKey(0)


# In[6]:


contours,hierarchy=cv2.findContours(edged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)    #finding contours
cv2.imshow("Canny Edges after Contouring",edged)
cv2.waitKey(0)


# In[7]:


print("Number of contours found= " +str(len(contours)))                     


# In[8]:


cv2.drawContours(image,contours,-1,(0,255,0),3)    #drawing all contours


# In[9]:


cv2.imshow("Contours",image)                       #displaying all contours
cv2.waitKey(0)


# In[10]:


cv2.destroyAllWindows()


# In[ ]:




