#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys                                  #import 'sys' module for system executables
print(sys.executable)


# In[2]:


import cv2                                  #importing necessary libraries
import numpy as np


# In[3]:


body_classifier=cv2.CascadeClassifier(cv2.samples.findFile(cv2.data.haarcascades+"haarcascade_frontalface_default.xml"))     #pointing to the haarcascade classifier function


# In[4]:


cap=cv2.VideoCapture("1.mov")              #capturing video           


# In[ ]:


while cap.isOpened():
    ret,frame=cap.read()
    rame=cv2.resize(frame,None,fx=0.5,fy=0.5,interpolation=cv2.INTER_LINEAR)
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    bodies=body_classifier.detectMultiScale(gray,1.2,3)
    
    for (x,y,w,h) in bodies:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)
        cv2.imshow("Pedestrians",frame)
    
    if cv2.waitKey(1)==3:
        break


# In[ ]:


cap.release()


# In[ ]:


cv2.destroyAllWindows()


# In[ ]:




